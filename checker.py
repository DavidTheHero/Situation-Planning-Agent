"""
Rail planner checker/simulator
-----------------------------
Cleaned and organized for readability:
- Grouped helpers and clear section headers
- Explicit constants for cargo types
- Guard-rails: no detaching loaded cargo at wrong towns; post-goal action detection
- Deadline lower-bounds and per-delivery enforcement hooks
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy
import re

from system_map import EDGE_HOURS, FACILITY, INITIAL, DUR, MAX_LOADED_CAPACITY
from problem_parser import parse_problem_description


# ---------- Module constants & exports ----------
__all__ = ["Checker", "check_plan", "SimulationError", "PlanError"]


CARGO_BOX = ("oranges", "bananas")
CARGO_LIQ = "juice"

# Small tolerance to avoid float round-off in time comparisons
EPS_TIME = 1e-6



# ========== Formatting & normalization helpers ==========

def hours_to_pretty(h: float) -> str:
    d, rem = divmod(h, 24)
    hours = int(rem)
    minutes = int((rem - hours) * 60)
    return f"{hours:02d}:{minutes:02d}" + (f" +{int(d)}d" if d >= 1 else "")

def mm_to_pretty(minutes: float) -> str:
    """Convert minutes to pretty HH:MM format with optional day suffix."""
    if minutes is None:
        return "00:00"
    return hours_to_pretty(minutes / 60.0)

def norm_cargo(x):
    if isinstance(x, list):
        # If it's a list, join or take the first element
        x = x[0] if x else ""
    x = (x or "").strip().lower()
    if x in ("oj", "orange juice", "o.j.", "o.j"):
        return "juice"
    return x

def is_loaded(car: Dict[str, str]) -> bool:
    return (car.get("type") == "boxcar" and car.get("payload") in CARGO_BOX) \
        or (car.get("type") == "tanker" and car.get("payload") == CARGO_LIQ)

def as_str(x) -> str:
    # Accept scalars or 1-element lists/tuples
    if isinstance(x, (list, tuple)):
        x = x[0] if x else ""
    return str(x)

def as_int(x) -> int:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else 0
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0
        
def _coerce_cargo_like(v, kind: str) -> str:
    """
    Normalize cargo/payload to a simple lower-case token:
    - Accepts scalars, 1-element lists/tuples, or dicts like {"oranges":"oranges"}.
    - Maps OJ variants to 'juice'.
    - For 'cargo', allow oranges/bananas only (LOAD writes boxcars).
    - For 'payload', allow oranges/bananas/juice/empty.
    """
    # unwrap list/tuple
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    # unwrap dict like {"oranges": "oranges"} -> "oranges"
    if isinstance(v, dict):
        if v:
            # prefer first key if it looks like a cargo word, else first value
            k = next(iter(v.keys()))
            if isinstance(k, str) and k.strip():
                v = k
            else:
                v = next(iter(v.values()))
        else:
            v = ""
    s = (str(v or "")).strip().lower()

    # normalize juice aliases
    if s in ("oj", "orange juice", "o.j.", "o.j"):
        s = "juice"

    if kind == "cargo":
        return s if s in ("oranges", "bananas") else s  # keep as-is but cleaned
    # payload
    if s not in ("oranges", "bananas", "juice", "empty"):
        # if something odd sneaks in, leave cleaned token; exec methods will guard
        return s
    return s

def _forbid_edges_from_constraints(constraints, start_hours=0):
        """Return a set of undirected edges to forbid at path-planning time based on constraints.
        Supports old schema ({"edge":[A,B], "start_h","end_h"}) and new schema ({"from","to","until"}).
        """
        forbid = set()
        for b in constraints.get("blocked_edges", []) or []:
            if "edge" in b:
                a, c = b.get("edge", [None, None])
                if not a or not c:
                    continue
                te = float(b.get("end_h", 0))
                if start_hours < te:
                    forbid.add(tuple(sorted((a, c))))
            else:
                a = b.get("from")
                c = b.get("to")
                te = b.get("until")
                if not a or not c or te is None:
                    continue
                try:
                    te = float(te)
                except Exception:
                    continue
                if start_hours < te:
                    forbid.add(tuple(sorted((a, c))))
        return forbid

# ========== Core simulator / checker ==========
class PlanError(Exception):
    step_idx: int
    detail: str
    def __str__(self) -> str:
        # matches your "s<idx>: ..." style
        return f"s{self.step_idx}: {self.detail}"
    
class SimulationError(Exception):
    pass

class Checker:
    """
    Simulator enforcing:
      - legal edges only; travel times from EDGE_HOURS (single-track)
      - capacity: ≤ 3 loaded cars per engine
      - facility legality (oranges@Corning, bananas@Avon, convert@Elmira)
      - CONVERT needs ≥1 oranges boxcar + ≥1 empty tanker attached; result: boxcar→empty, tanker→juice
      - WAIT at nodes only (we treat WAIT as time advance at current node)
      - time constraints:
          * deadline: all deliveries done by time
          * earliest arrival: forbid arrival at location before time
          * exact convert time at Elmira
          * leave immediately after convert (next step for that engine must be TRAVEL away)
    Plan schema: each step has {"act": ..., "args": {...}}.
      - When args include "engine", that engine is used.
      - For TRAVEL/LOAD without engine specified, we pick an engine at the 'from'/'at' node (if unique).
    """
    def __init__(self, problem_desc: str, goals: Dict[str, Dict[str, int]],
                 constraints: Dict[str, Any], overrides: Dict[str, Any]):
        # State
        self.loc_of_engine: Dict[str, str] = deepcopy(INITIAL["engines"])
        self.time_of_engine: Dict[str, float] = {eid: 0.0 for eid in self.loc_of_engine}
        # yard clocks per location (for warehouse work like LOAD without an engine)
        self.loc_time: Dict[str, float] = {loc: 0.0 for loc in FACILITY.keys()}
        self._start_order_by_loc: Dict[str, List[str]] = {loc: [] for loc in FACILITY.keys()}
        # Optional: preferred per-location start priority. Engines listed earlier get smaller offsets.
        # If not provided, default to prioritizing E2 at Elmira so E2 gets 0h and E3 gets +1h when both start there.
        self._start_priority_by_loc: Dict[str, List[str]] = dict(constraints.get("start_priority_by_loc", {}))
        self._start_priority_by_loc.setdefault("Elmira", ["E2", "E3"])  # default priority at Elmira
        # Pre-seed the order lists with the preferred priority (filtered to existing engines at init locations)
        for loc, prio in self._start_priority_by_loc.items():
            order = self._start_order_by_loc.setdefault(loc, [])
            for eid in prio:
                if eid in self.loc_of_engine and self.loc_of_engine[eid] == loc and eid not in order:
                    order.append(eid)
        self._start_stagger_hours: float = float(constraints.get("start_stagger_hours", 1.0) or 1.0)  # default 1 hour

        # Directional occupancy for edges where meets are allowed
        self.blocked_dir: dict[tuple[str, str], list[tuple[int, int]]] = {}

        # Allow meets (no opposite-direction conflicts) on specific undirected edges

        self.single_track_policy: str = (constraints or {}).get("single_track_policy", "blocked_only" if constraints.get("blocked_edges") else "off")
        self._single_track_disabled = False   # flip to True to ignore all blocking

        # arrival
        self.arrivals = {}

        self.constraints = constraints or {}

        txt = (problem_desc or "").lower()
        is_max_task = ("maximum" in txt or "determine the maximum" in txt or "max number" in txt)

        self.deadline_mode = self.constraints.get(
            "deadline_mode",
            "horizon" if is_max_task else "hard"
        )

        # yards: location -> list of cars dicts {"type": "boxcar"/"tanker", "payload": ...}
        self.yards: Dict[str, List[Dict[str, str]]] = {
            loc: []
            for loc in FACILITY.keys()
        }
        for loc, cars in INITIAL["boxcars"].items():
            for _ in cars:
                self.yards[loc].append({"type": "boxcar", "payload": "empty"})
        for loc, cars in INITIAL["tankers"].items():
            for _ in cars:
                self.yards[loc].append({"type": "tanker", "payload": "empty"})
        # consist per engine
        self.attached: Dict[str, List[Dict[str,str]]] = {eid: [] for eid in self.loc_of_engine}
        # --- Single-track reservations (directed) -------------------------
        # Directed reservations: key = (u, v), value = [(start_min, end_min)]
        self.blocked: Dict[Tuple[str,str], List[Tuple[int,int]]] = {}
        # Directional reservations for edges that allow meets/headways

        # Edges that allow opposite-direction meets (Elmira<->Corning)
        self.allow_meet_edges = { self._edge_key("Elmira", "Corning") }

        # Same-direction headway (hours) on special edges; 1.0 = allow trailing train if it departs ≥1h later
        self.same_dir_headway_edges = { self._edge_key("Elmira", "Corning"): 1.0 }

        # Default same-direction headway (hours). Override via constraints.
        self.default_same_dir_headway_hours: float = float(self.constraints.get("same_dir_headway_hours", 1.0) or 1.0)

        # Pre-block edges (apply to both directions). Support old and new schemas.
        for blk in self.constraints.get("blocked_edges", []) or []:
            if "edge" in blk:
                a, b = blk.get("edge", [None, None])
                start = float(blk.get("start_h", 0))
                end   = float(blk.get("end_h", 0))
            else:
                a = blk.get("from")
                b = blk.get("to")
                # default start from constraints.start_time (hours) or 0
                start = float(self.constraints.get("start_time", 0) or 0)
                end = float(blk.get("until", 0) or 0)
            if not a or not b:
                continue
            self.blocked.setdefault((a, b), []).append((start, end))
            self.blocked.setdefault((b, a), []).append((start, end))

        self.preblocked = deepcopy(self.blocked)

        # Overrides (prepositioned stock) -- dedupe with MAX per (type,payload,loc)
        agg = {}
        for ov in overrides.get("boxcars", []) or []:
            loc = ov["location"]
            payload = norm_cargo(ov.get("payload", "empty"))
            key = ("boxcar", payload, loc)
            agg[key] = max(agg.get(key, 0), int(ov.get("count", 0)))

        for ov in overrides.get("tankers", []) or []:
            loc = ov["location"]
            payload = norm_cargo(ov.get("payload", "empty"))  # allow 'juice'
            key = ("tanker", payload, loc)
            agg[key] = max(agg.get(key, 0), int(ov.get("count", 0)))

        for (typ, payload, loc), count in agg.items():
            for _ in range(count):
                self.yards[loc].append({"type": typ, "payload": payload})

        # Goals: cargo -> {dest: qty}
        # Normalize cargo names
        self.goals: Dict[str, Dict[str, int]] = {}
        # print(goals)
        for cargo, dests in (goals or {}).items():
            c = norm_cargo(cargo)
            self.goals.setdefault(c, {})
            if isinstance(dests, dict):
                for dest, qty in dests.items():
                    self.goals[c][dest] = int(qty)
            elif isinstance(dests, int):
                # fallback if parser ever returns an int without dest
                self.goals[c]["Elmira"] = int(dests)
        
        # If no goals parsed, try to derive from the problem description
        if not self.goals:
            inferred = self._derive_goals_from_desc(problem_desc or "")
            if inferred:
                self.goals = inferred

        # Constraints
        # deadline: hours; earliest arrival can be a single time (applies to Avon) or dict {loc: time}
        self.deadline: Optional[float] = constraints.get("deadline")
        # allow both shapes:
        ea = constraints.get("earliest_arrival")
        self.earliest_arrivals: Dict[str, float] = constraints.get("earliest_arrivals", {})
        if isinstance(ea, (int, float)):
            # If parser didn't capture the location, assume it's Avon (common case in problems).
            self.earliest_arrivals.setdefault("Avon", float(ea))
        self.exact_convert_time: Optional[float] = constraints.get("exact_convert_time")
        self.leave_after_convert: bool = bool(constraints.get("leave_after_convert"))

        # Track immediate-leave enforcement: engine -> must_depart_from (str) or None
        self.must_depart_from: Dict[str, Optional[str]] = {eid: None for eid in self.loc_of_engine}

        self.problem_desc = problem_desc
        self.violations: List[Any] = []
        self.violation_details: List[Dict[str, Any]] = []


    # ----- Utility

    # --- JSON-safe utilities ---------------------------------

    def _json_safe_key(self, k):
        # Only for DICT KEYS
        if isinstance(k, tuple):
            try:
                return f"{k[0]}<->{k[1]}"
            except Exception:
                return str(k)
        return k

    def _json_safe(self, obj):
        # Recursively sanitize KEYS and VALUES
        if isinstance(obj, dict):
            return { self._json_safe_key(k): self._json_safe(v) for k, v in obj.items() }
        if isinstance(obj, (list, tuple)):
            return [ self._json_safe(x) for x in obj ]
        return obj

    # ---------- Structured violation helpers ----------
    def _count_attached(self, eid: str, typ: str, payload: str) -> int:
        if not eid or eid not in self.attached:
            return 0
        return sum(1 for c in self.attached[eid] if c["type"] == typ and c["payload"] == payload)

    def _count_yard(self, loc: str, typ: str, payload: str) -> int:
        if not loc:
            return 0
        return sum(1 for c in self.yards.get(loc, []) if c["type"] == typ and c["payload"] == payload)

    def _nearest_sources_from(self, origin: str, typ: str, payload: str, k: int = 3):
        """
        List up to k yards that currently have (typ,payload), sorted by hours from origin.
        Returns items like: (hours, loc, count, path)
        """
        out = []
        for loc in FACILITY.keys():
            cnt = self._count_yard(loc, typ, payload)
            if cnt <= 0:
                continue
            path = self.shortest_path(origin, loc)
            if not path or len(path) < 2:
                continue
            out.append((self.path_cost(path), loc, cnt, path))
        out.sort(key=lambda t: t[0])
        return out[:k]

    def _post_attach_balance_note(self, loc: str, typ: str, payload: str, take: int) -> str:
        have = self._count_yard(loc, typ, payload)
        remain = max(0, have - max(0, int(take)))
        return f"(after ATTACH {take} at {loc}, yard {typ}({payload}) will be {remain})"

    def _yard_counts(self, loc: str) -> Dict[str, int]:
        out = {"empty_boxcars": 0, "empty_tankers": 0, "oranges": 0, "bananas": 0, "juice": 0}
        for c in self.yards.get(loc, []):
            if c["type"] == "boxcar" and c["payload"] == "empty":   out["empty_boxcars"] += 1
            if c["type"] == "tanker" and c["payload"] == "empty":   out["empty_tankers"] += 1
            if c["type"] == "boxcar" and c["payload"] == "oranges": out["oranges"] += 1
            if c["type"] == "boxcar" and c["payload"] == "bananas": out["bananas"] += 1
            if c["type"] == "tanker" and c["payload"] == "juice":   out["juice"] += 1
        return out

    def _engine_snapshot(self, eid: Optional[str]) -> Dict[str, Any]:
        if not eid or eid not in self.loc_of_engine:
            return {}
        loc = self.loc_of_engine[eid]
        return {
            "engine": eid,
            "loc": loc,
            "time_hours": self.time_of_engine[eid],
            "time_pretty": hours_to_pretty(self.time_of_engine[eid]),
            "attached": [dict(c) for c in self.attached[eid]],
            "yard_at_loc": self._yard_counts(loc)
        }

    def _classify_rule(self, act: str, msg: str, args: Dict[str, Any]) -> str:
        m = msg.lower()
        if "exceeds loaded capacity" in m: return "capacity"
        if m.startswith("travel: illegal edge"): return "illegal-edge"
        if "occupied" in m and "travel" in m: return "single-track"
        if "not at" in m and act in ("ATTACH","DETACH","TRAVEL","UNLOAD"): return "engine-location"
        if m.startswith("attach: not enough"): return "attach-availability"
        if m.startswith("load: not enough empty boxcars"): return "load-empties"
        if m.startswith("load:") and "oranges" in m and args.get("at") != "Corning": return "load-site"
        if m.startswith("load:") and "bananas" in m and args.get("at") != "Avon": return "load-site"
        if m.startswith("convert: allowed only at"): return "convert-site"
        if m.startswith("convert: need loaded oranges"): return "convert-prereq"
        if m.startswith("deadline"): return "deadline"
        if m.startswith("travel: arrived at") and "before" in m: return "earliest"
        if m.startswith("goal: need"): return "goal-shortfall"
        if m.startswith("unknown act"): return "act-unknown"
        return "generic"

    def _nearest_yard_with(self, predicate) -> Optional[Tuple[str, int]]:
        """Return (loc, hours) of nearest yard satisfying predicate on current yard inventory."""
        best = None
        for loc in FACILITY.keys():
            if predicate(self._yard_counts(loc)):
                # Use Elmira's clock as reference for path cost; this is for hinting only
                # Better: choose the most relevant engine or a caller-provided node.
                # We’ll pick the most idle engine (lowest time) as origin.
                origin = min(self.loc_of_engine, key=lambda e: self.time_of_engine[e])
                src = self.loc_of_engine[origin]
                path = self.shortest_path(src, loc) or []
                if not path: 
                    continue
                hours = self.path_cost(path)
                if best is None or hours < best[1]:
                    best = (loc, hours)
        return best

    def _hint_for(self, rule: str, act: str, args: Dict[str, Any], msg: str) -> str:
        rule = (rule or "").lower()
        act  = (act  or "").upper()

        # -------- CONVERT prereq failures --------
        if rule == "convert-prereq":
            eid = args.get("engine")
            qty = int(args.get("qty", 1) or 1)
            here = self.loc_of_engine.get(eid, "")
            have_or = self._count_attached(eid, "boxcar", "oranges")
            have_tk = self._count_attached(eid, "tanker", "empty")
            miss_or = max(0, qty - have_or)
            miss_tk = max(0, qty - have_tk)

            parts = []
            parts.append(f"Make convert_ready for {eid}: need {qty} oranges-boxcar(s) + {qty} empty tanker(s) attached at Elmira.")
            parts.append(f"Currently attached: oranges={have_or}, empty_tankers={have_tk}; missing: oranges={miss_or}, tankers={miss_tk}.")

            # Missing oranges → fetch empty boxcars (nearest to current engine), then LOAD at Corning
            if miss_or > 0:
                src_or = self._nearest_sources_from(here, "boxcar", "empty", k=1)
                if src_or:
                    h, loc, cnt, path = src_or[0]
                    take = min(miss_or, cnt)
                    parts.append(
                        f"For oranges: ATTACH {take} empty boxcar(s) at {loc} "
                        f"{self._post_attach_balance_note(loc,'boxcar','empty',take)}; "
                        f"TRAVEL to Corning via {'→'.join(path)} (≈{h}h) and LOAD oranges (1h)."
                    )
                else:
                    parts.append("For oranges: no yard has empty boxcars; free attached empties or reduce qty.")

            # Missing tankers → attach at Corning (or closest yard that has empty tankers)
            if miss_tk > 0:
                src_tk = self._nearest_sources_from(here, "tanker", "empty", k=1)
                if src_tk:
                    h, loc, cnt, path = src_tk[0]
                    take = min(miss_tk, cnt)
                    travel_note = "" if loc == "Corning" else f"TRAVEL to Corning via {'→'.join(path)} (≈{h}h), then "
                    parts.append(
                        f"For tankers: {travel_note}ATTACH {take} empty tanker(s) at Corning "
                        f"{self._post_attach_balance_note('Corning','tanker','empty',take)}."
                    )
                else:
                    parts.append("For tankers: no empty tankers in yards; adjust plan or reduce qty.")

            parts.append("Then TRAVEL to Elmira and CONVERT (1h).")
            return " ".join(parts)

        # -------- LOAD short of empties at site --------
        if rule == "load-empties":
            eid  = args.get("engine")
            at   = args.get("at") or ""
            cars = int(args.get("cars", 0) or 0)
            attached_empties = self._count_attached(eid, "boxcar", "empty")
            yard_empties     = self._count_yard(at, "boxcar", "empty")
            miss = max(0, cars - (attached_empties + yard_empties))

            parts = []
            parts.append(f"To LOAD {cars} at {at}: attached_empties={attached_empties}, yard_empties={yard_empties}, missing={miss}.")
            if miss <= 0:
                parts.append("ATTACH empties here (no detour), then LOAD (1h).")
            else:
                nearby = self._nearest_sources_from(at, "boxcar", "empty", k=1)
                if nearby:
                    h, loc, cnt, path = nearby[0]
                    take = min(miss, cnt)
                    parts.append(
                        f"Fetch empties: TRAVEL to {loc} via {'→'.join(path)} (≈{h}h), "
                        f"ATTACH {take} empty boxcar(s) {self._post_attach_balance_note(loc,'boxcar','empty',take)}, "
                        f"TRAVEL back to {at}, then LOAD (1h)."
                    )
                else:
                    parts.append("No yard has spare empties; free some or lower the cars count.")
            return " ".join(parts)

        # -------- Wrong LOAD site --------
        if rule == "load-site":
            cargo = norm_cargo(args.get("cargo"))
            where = "Corning" if cargo == "oranges" else ("Avon" if cargo == "bananas" else "?")
            return f"Load {cargo} only at {where}. TRAVEL to {where} before LOAD."

        # -------- Engine not at required node --------
        if rule == "engine-location":
            loc = args.get("from") or args.get("at") or args.get("to") or "the required node"
            return f"TRAVEL to {loc} first, then retry {act}."

        # -------- Track occupancy --------
        if rule in ("single-track","edge-occupied") or ("occupied" in (msg or "").lower()):
            frm = (args.get("from") or "")
            to  = (args.get("to") or "")
            path = self.shortest_path(frm, to)
            dur  = self.path_cost(path) if path else None
            base = f"Edge {frm}↔{to} is occupied now. Options: WAIT until free, then TRAVEL"
            if path and len(path) > 2:
                base += f"; or take alternate route {'→'.join(path)} (≈{dur}h)."
            else:
                base += "."
            return base

        # -------- Deadline --------
        if rule == "deadline":
            return "Reduce detours and idle time; split loads across engines; keep per-engine loaded ≤3; use shortest paths and overlap trips."

        # -------- Goal shortfall (cargo-specific mini-recipe) --------
        if rule == "goal-shortfall":
            cargo = (args or {}).get("cargo")
            dest  = (args or {}).get("dest")
            need  = int((args or {}).get("need", 0) or 0)
            have  = int((args or {}).get("have", 0) or 0)
            remain = max(0, need - have)

            if cargo == "oranges" and dest == "Bath":
                cor = self._yard_counts("Corning")["oranges"]
                cap_hint = "Respect per-engine loaded ≤3; use multiple engines if remaining >3."
                if cor > 0:
                    return (f"Remaining: {remain} orange boxcar(s) to Bath. Corning has {cor} loaded. "
                            f"Start E3 (Elmira) → Corning → ATTACH up to 3 loaded → Bath. "
                            f"If still short, start E1 (Avon) → Dansville → Corning → ATTACH remaining → Bath. "
                            f"{cap_hint}")
            return "Plan additional trip(s) for the remaining units; prefer shortest path and nearest empties."

        # -------- Fallback --------
        return "Fix preconditions at the current node first; ATTACH → (TRAVEL) → LOAD/CONVERT → TRAVEL to destination; keep per-engine loaded ≤3."
    
    def _record_violation(self, step_idx: int, act: str, args: Dict[str, Any], msg: str):
        # 1) Legacy string (kept for backward-compat) — now enriched with rule and hint for downstream hint extractors
        # We construct the structured detail first to compute the hint, then append a richer legacy string.
        args = args or {}
        eid = args.get("engine")
        rule = self._classify_rule(act, msg, args)
        detail = {
            "step": step_idx,
            "rule": rule,
            "act": act,
            "message": msg,
            "args": dict(args),  # values may include tuples; we sanitize below
            "time_pretty": hours_to_pretty(self.time_of_engine.get(eid, 0)) if eid in self.time_of_engine else None,
            "engine": self._engine_snapshot(eid),
            "yard_at_from": self._yard_counts(args.get("from")) if args.get("from") else None,
            "yard_at_at": self._yard_counts(args.get("at")) if args.get("at") else None,
            "yard_at_to": self._yard_counts(args.get("to")) if args.get("to") else None,
            "now_hours": self.time_of_engine.get(eid, None) if eid in self.time_of_engine else None,
        }
        hint = self._hint_for(rule, act, args, msg)
        if hint:
            detail["hint"] = hint
            
        # Propagate any structured suggestions added by executors (e.g., nearest yard fetch)
        if isinstance(args, dict) and "suggestion" in args:
            try:
                detail["suggestion"] = self._json_safe(args["suggestion"])  # ensure JSON-safe
            except Exception:
                detail["suggestion"] = args.get("suggestion")
        # Enrich legacy string to include rule + hint so text-based extractors can see it
        legacy = f"s{step_idx}: {msg}"
        if rule:
            legacy += f" | rule: {rule}"
        if hint:
            legacy += f" | hint: {hint}"
        self.violations.append(legacy)

        # 2) Structured detail (JSON-safe)
        # Ensure fully JSON-safe (no tuple keys; tuple/list values ok after conversion)
        if not hasattr(self, "violation_details"):
            self.violation_details = []
        self.violation_details.append(self._json_safe(detail))


    def _normalize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (args or {}).items():
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None

            if k in ("qty","cars","hours"):
                out[k] = 0 if v is None else (int(v) if isinstance(v,(int,float)) or (isinstance(v,str) and v.isdigit()) else int(float(v)))
            elif k in ("cargo", "payload"):
                out[k] = _coerce_cargo_like(v, k)
            elif k in ("engine","from","to","at","car_type","until"):
                out[k] = None if v is None else str(v)
            else:
                out[k] = v
        return out



    def _dir_key(self, a: str, b: str) -> Tuple[str,str]:
        return (a, b)
    
    def _edge_key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)


    def _occupy_edge(self, a: str, b: str, start_hours: float, dur_h: float) -> bool:
        edge_dir = (a, b)  # Use directional key to match how blocked edges are stored
        edge_undirected = self._edge_key(a, b)
        end  = start_hours + dur_h

        policy = self.single_track_policy

        # Policy OFF: never block
        if policy == "off":
            return True

        # Policy BLOCKED_ONLY: only check against preblocked windows; do NOT add dynamic reservations
        if policy == "blocked_only":
            # Check both directions for preblocked edges
            for s, e in self.preblocked.get(edge_dir, []):
                if not (end <= s or start_hours >= e):
                    return False
            return True

        # Policy FULL: enforce dynamic conflicts across all trains (current behavior)
        segs = self.blocked.setdefault(edge_undirected, [])
        for s, e in segs:
            if not (end <= s or start_hours >= e):
                return False
        segs.append((start_hours, end))
        return True

    
    def _occupy_edge_dir(self, a: str, b: str, start_hours: float, dur_h: float) -> bool:
        if getattr(self, "_single_track_disabled", False):
            return True

        return self._occupy_edge(a, b, start_hours, dur_h)




    def _choose_engine_at(self, loc: str) -> Optional[str]:
        eids = [eid for eid, l in self.loc_of_engine.items() if l == loc]
        if not eids:
            return None
        if len(eids) > 1:
            # ambiguous — for now pick the earliest-available (smallest clock) to keep it simple
            eids.sort(key=lambda eid: self.time_of_engine[eid])
        return eids[0]

    def _capacity_ok(self, eid: str) -> bool:
        loaded = sum(1 for c in self.attached[eid] if is_loaded(c))
        return loaded <= MAX_LOADED_CAPACITY

    def _require_capacity(self, eid: str, ctx: str):
        if not self._capacity_ok(eid):
            raise SimulationError(f"{ctx}: Engine {eid} exceeds loaded capacity {MAX_LOADED_CAPACITY}")
    
    def edge_hours(self, a: str, b: str) -> int:
        """Undirected edge lookup; raises if missing."""
        h = EDGE_HOURS.get((a, b))
        if h is None:
            h = EDGE_HOURS.get((b, a))
        if h is None:
            raise SimulationError(f"TRAVEL: illegal edge {a}->{b}")
        return h

    def path_cost(self, path: List[str]) -> int:
        """Sum hours along a path list like ['Elmira','Corning','Dansville']."""
        if not path or len(path) == 1:
            return 0
        total = 0
        for u, v in zip(path, path[1:]):
            total += self.edge_hours(u, v)
        return total

    # ========== Action executors ==========

    def _exec_START(self, args: Dict[str, Any]):
        eid = args.get("engine")
        at = args.get("at")
        if eid not in self.loc_of_engine:
            raise SimulationError(f"START: unknown engine {eid}")
        if self.loc_of_engine[eid] != at:
            raise SimulationError(f"START: engine {eid} not at {at} (is at {self.loc_of_engine[eid]})")

        # Per-location auto-stagger: the Nth unique START at this 'at' sets clock to (N-1)*stagger
        order = self._start_order_by_loc.setdefault(at, [])
        if eid not in order:
            order.append(eid)
            offset_hours = (len(order) - 1) * self._start_stagger_hours
            # Only move forward in time; never rewind an engine's clock
            if self.time_of_engine[eid] < offset_hours:
                self.time_of_engine[eid] = offset_hours
        # no time cost beyond the stagger

    def _exec_ATTACH(self, args):
        eid = args.get("engine")
        car_type = as_str(args.get("car_type")).lower()
        qty = as_int(args.get("qty", 0))
        loc = as_str(args.get("from"))
        want_payload = as_str(args.get("payload")).lower() if args.get("payload") is not None else ""

        if eid not in self.loc_of_engine:
            raise SimulationError("ATTACH: unknown engine")
        if self.loc_of_engine[eid] != loc:
            raise SimulationError(f"ATTACH: engine {eid} not at {loc}")

        picked = 0
        yard = self.yards[loc]
        j = 0
        while j < len(yard) and picked < qty:
            c = yard[j]
            if c["type"] == car_type and (not want_payload or c["payload"] == want_payload):
                self.attached[eid].append(yard.pop(j))
                picked += 1
            else:
                j += 1

        if picked < qty:
            have = sum(
                1 for c in yard
                if c["type"] == car_type and (not want_payload or c["payload"] == want_payload)
            )
            # Enhance violation with nearest yard suggestion
            suggestion = ""
            available = have
            if available < qty:
                # Try to find nearest yard with available cars of same type
                min_dist = None
                nearest = None
                for yard_loc, cars in self.yards.items():
                    if yard_loc == loc:
                        continue
                    count = sum(
                        1 for c in cars
                        if c.get("type") == car_type and not is_loaded(c)
                    )
                    if count >= qty:
                        try:
                            dist = self._shortest_path_time(loc, yard_loc)
                        except Exception:
                            continue
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            nearest = yard_loc
                if nearest:
                    suggestion = f" Consider fetching from nearest yard with availability: {nearest} ({min_dist} hours away)."
                    # also attach a machine-readable suggestion for the violation details
                    try:
                        args.setdefault("suggestion", {})["fetch"] = {
                            "car_type": car_type,
                            "payload": (want_payload or "any"),
                            "from": nearest,
                            "hours": float(min_dist),
                            "qty": int(qty),
                        }
                    except Exception:
                        pass
            if want_payload:
                if have == 0:
                    raise SimulationError(f"ATTACH: no {car_type}s with {want_payload} at {loc} (need {qty}).{suggestion}")
                else:
                    raise SimulationError(
                        f"ATTACH: not enough {car_type}s with {want_payload} at {loc} (have {have}, need {qty}).{suggestion}"
                    )
            else:
                if have == 0:
                    raise SimulationError(f"ATTACH: no {car_type}s at {loc} (need {qty}).{suggestion}")
                else:
                    raise SimulationError(
                        f"ATTACH: not enough {car_type}s at {loc} (have {have}, need {qty}).{suggestion}"
                    )

        # capacity check (only loaded cars count)
        self._require_capacity(eid, "ATTACH")

    def _shortest_path_time(self, a: str, b: str) -> float:
        """
        Returns the shortest path time (in hours) from a to b using edge hours.
        """
        path = self.shortest_path(a, b)
        if not path or len(path) < 2:
            raise Exception("No path found")
        return float(self.path_cost(path))

    def _exec_DETACH(self, args: Dict[str, Any]):
        eid = args.get("engine")
        qty = as_int(args.get("qty", 0))
        car_type = as_str(args.get("car_type")).lower()
        at = as_str(args.get("at"))
        want_payload = as_str(args.get("payload")).lower() if args.get("payload") is not None else ""

        if eid not in self.loc_of_engine:
            raise SimulationError("DETACH: unknown engine")
        if self.loc_of_engine[eid] != at:
            raise SimulationError(f"DETACH: engine {eid} not at {at}")

        took = 0
        i = 0
        detached: list[dict] = []
        while i < len(self.attached[eid]) and took < qty:
            c = self.attached[eid][i]
            if c["type"] == car_type and (not want_payload or c["payload"] == want_payload):
                detached.append(self.attached[eid].pop(i))
                took += 1
            else:
                i += 1

        if took < qty:
            if want_payload:
                raise SimulationError(f"DETACH: not enough {car_type}(s) with {want_payload} attached")
            else:
                raise SimulationError(f"DETACH: not enough {car_type}(s) attached")

        # Forbid detaching loaded cargo at a non-destination town
        for c in detached:
            payload = c.get("payload")
            if c.get("type") == "boxcar" and payload in CARGO_BOX:
                valid_dests = set((self.goals.get(payload, {}) or {}).keys())
                if valid_dests and at not in valid_dests:
                    raise SimulationError(
                        f"DETACH: cannot drop loaded {payload} at {at}; required destination(s): {sorted(valid_dests)}"
                    )
            if c.get("type") == "tanker" and c.get("payload") == CARGO_LIQ:
                valid_dests = set((self.goals.get(CARGO_LIQ, {}) or {}).keys())
                if valid_dests and at not in valid_dests:
                    raise SimulationError(
                        f"DETACH: cannot drop loaded {CARGO_LIQ} at {at}; required destination(s): {sorted(valid_dests)}"
                    )

        # Move the detached cars to the yard after validation
        for c in detached:
            self.yards[at].append(c)

        # capacity check (only loaded cars count)
        self._require_capacity(eid, "DETACH")

    def _exec_LOAD(self, args: Dict[str, Any]):
        at    = as_str(args.get("at"))
        cargo = norm_cargo(args.get("cargo"))
        cars  = as_int(args.get("cars", 0))

        # normalize/select engine
        eid = args.get("engine")
        if isinstance(eid, (list, tuple)):
            eid = eid[0] if eid else None
        if eid is not None:
            eid = as_str(eid)
        if eid is None:
            eid = self._choose_engine_at(at)

        # If an engine is specified, require it to actually be at 'at'
        if args.get("engine") is not None and self.loc_of_engine.get(eid) != at:
            raise SimulationError(f"LOAD: engine {eid} not at {at} (is at {self.loc_of_engine.get(eid)})")

        attached_loaded = 0
        if eid and self.loc_of_engine.get(eid) == at:
            empties_idx = [i for i,c in enumerate(self.attached[eid])
                        if c["type"]=="boxcar" and c["payload"]=="empty"]
            take = min(cars, len(empties_idx))
            for k in range(take):
                self.attached[eid][empties_idx[k]]["payload"] = cargo
            attached_loaded = take
            if attached_loaded > 0:
                self.time_of_engine[eid] += DUR["LOAD"]
                self._require_capacity(eid, "LOAD")

        remaining = cars - attached_loaded
        if remaining <= 0:
            return

        # load empties from yard (no engine required)
        yard = self.yards[at]
        loaded_from_yard = 0
        j = 0
        while j < len(yard) and loaded_from_yard < remaining:
            if yard[j]["type"] == "boxcar" and yard[j]["payload"] == "empty":
                yard[j]["payload"] = cargo
                loaded_from_yard += 1
                j += 1
            else:
                j += 1

        if loaded_from_yard < remaining:
            # rich diagnostics
            attached_here = 0
            if eid and self.loc_of_engine.get(eid) == at:
                attached_here = sum(1 for c in self.attached[eid]
                                    if c["type"]=="boxcar" and c["payload"]=="empty")
            yard_empty = sum(1 for c in self.yards[at]
                            if c["type"]=="boxcar" and c["payload"]=="empty")
            # Suggest the nearest yard with empty boxcars to fetch from
            try:
                nearby = self._nearest_sources_from(at, "boxcar", "empty", k=1)
            except Exception:
                nearby = []
            if nearby:
                h, loc, cnt, path = nearby[0]
                try:
                    args.setdefault("suggestion", {})["fetch"] = {
                        "car_type": "boxcar",
                        "payload": "empty",
                        "from": loc,
                        "hours": float(h),
                        "qty": int(remaining),
                    }
                except Exception:
                    pass
                raise SimulationError(
                    f"LOAD: need {cars} empty boxcars at {at}; have attached={attached_here}, "
                    f"yard={yard_empty}. Consider fetching from {loc} (≈{h}h)."
                )
            else:
                raise SimulationError(
                    f"LOAD: need {cars} empty boxcars at {at}; have attached={attached_here}, yard={yard_empty}"
                )

        # yard work consumes 1h (once)
        self.loc_time[at] += DUR["LOAD"]



    def _exec_UNLOAD(self, args: Dict[str, Any]):
        at = as_str(args.get("at"))
        cargo = norm_cargo(args.get("cargo"))
        cars = as_int(args.get("cars", 0))
        eid = args.get("engine") or self._choose_engine_at(at)
        if not eid:
            raise SimulationError(f"UNLOAD: no engine at {at}")
        loaded = [i for i,c in enumerate(self.attached[eid]) if c["type"]=="boxcar" and c["payload"]==cargo]
        if len(loaded) < cars:
            raise SimulationError(f"UNLOAD: not enough {cargo} attached")
        for i in range(cars):
            self.attached[eid][loaded[i]]["payload"] = "empty"
        self.time_of_engine[eid] += DUR["UNLOAD"]

    def _exec_CONVERT(self, args: Dict[str, Any]):
        at  = as_str(args.get("at", "Elmira"))
        qty = as_int(args.get("qty", 1))

        # normalize engine early
        eid = args.get("engine") or self._choose_engine_at(at)
        if isinstance(eid, (list, tuple)):
            eid = eid[0] if eid else None
        if eid is not None:
            eid = as_str(eid)
        if not eid:
            raise SimulationError(f"CONVERT: no engine at {at}")
        if at != "Elmira":
            raise SimulationError("CONVERT: allowed only at Elmira")

        # debug snapshot before gating
        # debug = [(c["type"], c["payload"]) for c in self.attached[eid]]
        # print(f"[DEBUG] CONVERT precheck @ {mm_to_pretty(self.time_of_engine[eid])} on {eid} attached={debug}, qty={qty}", flush=True)

        # exact-time constraint (optional)
        if self.exact_convert_time is not None:
            if self.time_of_engine[eid] != self.exact_convert_time:
                raise SimulationError(
                    f"CONVERT: must start at {hours_to_pretty(self.exact_convert_time)} "
                    f"(now {hours_to_pretty(self.time_of_engine[eid])})"
                )

        # readiness from current attached
        oranges_idx = [i for i,c in enumerate(self.attached[eid]) if c["type"]=="boxcar" and c["payload"]=="oranges"]
        empty_idx   = [i for i,c in enumerate(self.attached[eid]) if c["type"]=="tanker" and c["payload"]=="empty"]
        # print(f"[DEBUG] CONVERT readiness: oranges={len(oranges_idx)}, empty_tankers={len(empty_idx)}, need={qty}", flush=True)

        if len(oranges_idx) < qty or len(empty_idx) < qty:
            snap = [(c["type"], c["payload"]) for c in self.attached[eid]]
            raise SimulationError(
                f"CONVERT: need loaded oranges boxcar(s) and empty tanker(s) attached; "
                f"have oranges={len(oranges_idx)}, tankers={len(empty_idx)}; attached={snap}"
            )

        # do the conversion
        for k in range(qty):
            self.attached[eid][oranges_idx[k]]["payload"] = "empty"
            self.attached[eid][empty_idx[k]]["payload"]   = "juice"
        self.time_of_engine[eid] += DUR["CONVERT"]
        # print(f"[DEBUG] CONVERT applied: attached={[(c['type'], c['payload']) for c in self.attached[eid]]}", flush=True)

        if self.leave_after_convert:
            self.must_depart_from[eid] = at


    def _exec_TRAVEL(self, args: Dict[str, Any]):
        frm = as_str(args.get("from"))
        to = as_str(args.get("to"))
        eid = args.get("engine") or self._choose_engine_at(frm)
        if not eid:
            raise SimulationError(f"TRAVEL: no engine at {frm}")
        if self.loc_of_engine[eid] != frm:
            raise SimulationError(f"TRAVEL: engine {eid} not at {frm} (is at {self.loc_of_engine[eid]})")

        # No-op travel violation
        if frm == to:
            raise SimulationError(f"TRAVEL: illegal edge {frm}->{to} (no-op travel)")

        dur_h = self.edge_hours(frm, to)
        self._require_capacity(eid, "TRAVEL")

        start = self.time_of_engine[eid]
        if not self._occupy_edge(frm, to, start, dur_h):
            raise SimulationError(f"TRAVEL: edge {frm}<->{to} occupied at {hours_to_pretty(start)}")

        # move
        self.time_of_engine[eid] += dur_h
        self.loc_of_engine[eid] = to
        # log arrivals for loaded cars reaching 'to'
        arr_t = self.time_of_engine[eid]
        for c in self.attached[eid]:
            if c["type"] == "boxcar" and c["payload"] in CARGO_BOX:
                key = (c["payload"], to)
                self.arrivals.setdefault(key, []).append(arr_t)
            if c["type"] == "tanker" and c["payload"] == CARGO_LIQ:
                key = (CARGO_LIQ, to)
                self.arrivals.setdefault(key, []).append(arr_t)
        # earliest arrival?
        ea = self.earliest_arrivals.get(to)
        if ea is not None and self.time_of_engine[eid] < ea:
            raise SimulationError(f"TRAVEL: arrived at {to} before {mm_to_pretty(ea)}")
        # if we were required to leave immediately after convert, clear that requirement only if we left the marked node
        if self.must_depart_from[eid] == frm:
            self.must_depart_from[eid] = None  # departed

    def _exec_WAIT(self, args: Dict[str, Any]):
        # WAIT formats accepted:
        #   {"minutes": 90}  or  {"hours": 3}  or  {"until": "+3h"}
        eid = args.get("engine")
        if eid is None:
            # pick the engine with the earliest current time
            eid = min(self.time_of_engine.keys(), key=lambda k: self.time_of_engine[k])
        inc = 0.0
        if "hours" in args: 
            hours_val = args["hours"]
            if isinstance(hours_val, (int, float)):
                inc = float(hours_val)
            else:
                inc = float(str(hours_val))
        elif "until" in args:
            m = re.match(r"\+(\d+(?:\.\d+)?)h", as_str(args["until"]).strip().lower())
            if not m:
                raise SimulationError("WAIT: 'until' must look like '+Nh' or '+N.Mh'")
            inc = float(m.group(1))
        else:
            raise SimulationError("WAIT: need hours/until")
        # cannot wait at Elmira immediately after convert (if constraint present)
        if self.leave_after_convert and self.must_depart_from.get(eid) == self.loc_of_engine[eid]:
            raise SimulationError("WAIT: must leave Elmira immediately after convert")
        self.time_of_engine[eid] += inc


# ========== Plan execution ==========
    def run(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # ---------- NEW: per-delivery deadline infeasibility check (lower bounds) ----------
        # Build a per-delivery deadline map. If only a global deadline exists, mirror it to each (cargo,dest) goal.
        per_dead = dict(self.constraints.get("deliver_deadlines", {}))  # {"cargo": {"Loc": minutes}}
        if not per_dead and self.deadline is not None:
            for cargo, dests in self.goals.items():
                for dest in dests:
                    per_dead.setdefault(cargo, {})[dest] = self.deadline

        if per_dead:
            for cargo, locs in per_dead.items():
                for loc, t_dead in (locs or {}).items():
                    need = int(self.goals.get(cargo, {}).get(loc, 0))
                    if need <= 0:
                        continue  # nothing required there
                    lb_hours = self._lb_one_unit_hours(cargo, loc)
                    if lb_hours == float("inf"):
                        self._record_violation(
                            0, "DEADLINE",
                            {"cargo": cargo, "dest": loc, "need": need, "deadline_hours": float(t_dead)},
                            f"Deadline infeasible: cannot route/load 1 {cargo} to {loc} with current network/stock (deadline {hours_to_pretty(float(t_dead))})."
                        )
                    elif lb_hours is not None and lb_hours > float(t_dead) + EPS_TIME:
                        self._record_violation(
                            0, "DEADLINE",
                            {
                                "cargo": cargo, "dest": loc, "need": need,
                                "lower_bound_hours": float(lb_hours), "deadline_hours": float(t_dead)
                            },
                            f"Deadline infeasible: need ≥ {hours_to_pretty(float(lb_hours))} for 1 {cargo}→{loc}, but deadline is {hours_to_pretty(float(t_dead))}."
                        )
        if not self.goals and plan.get("goals"):
            self.goals = {norm_cargo(k): {loc:int(v) for loc,v in d.items()} for k,d in plan["goals"].items()}
        steps = plan.get("steps", [])
        first_goal_met_idx = None  # 1-based index when goals first become satisfied
        for i, st in enumerate(steps, start=1):
            act = st.get("act")
            args = self._normalize_args(st.get("args", {}) or {})
            try:
                if act == "START":    self._exec_START(args)
                elif act == "ATTACH": self._exec_ATTACH(args)
                elif act == "DETACH": self._exec_DETACH(args)
                elif act == "LOAD":   self._exec_LOAD(args)
                elif act == "UNLOAD": self._exec_UNLOAD(args)
                elif act == "CONVERT":self._exec_CONVERT(args)
                elif act == "TRAVEL": self._exec_TRAVEL(args)
                elif act == "WAIT":   self._exec_WAIT(args)
                else:
                    raise SimulationError(f"Unknown act {act}")
                # After applying this step, check if all goals are satisfied for the first time
                if first_goal_met_idx is None and self.goals_met():
                    first_goal_met_idx = i
            except SimulationError as e:
                self._record_violation(i, act or "", args, str(e))

        # Violation: post-goal actions after goals were already satisfied
        if first_goal_met_idx is not None and first_goal_met_idx < len(steps):
            extra = len(steps) - first_goal_met_idx
            self._record_violation(
                first_goal_met_idx + 1,
                "POST",
                {},
                f"Post-goal actions present: {extra} extra step(s) after all goals were satisfied"
            )

        # After steps, evaluate goals & deadline
        goal_achieved = self._check_goals()
        # Per-delivery deadlines (if any)
        deliver_deadlines = self.constraints.get("deliver_deadlines", {})

        
        if deliver_deadlines:
            for cargo, locs in deliver_deadlines.items():
                for loc, t_dead in locs.items():
                    need = self.goals.get(cargo, {}).get(loc, 0)
                    if need <= 0:
                        continue  # no required units => nothing to enforce
                    arr_times = sorted(self.arrivals.get((cargo, loc), []))
                    on_time = sum(1 for tt in arr_times if tt <= t_dead + EPS_TIME)
                    if on_time < need:
                        self._record_violation(
                            0, "DEADLINE",
                            {"cargo": cargo, "dest": loc, "need": need, "deadline_hours": t_dead},
                            f"Deadline for {need} {cargo} at {loc} is {hours_to_pretty(t_dead)} (on-time arrivals {on_time}/{need})"
                        )
                        goal_achieved = False
        else:
            # No per-delivery deadlines parsed; decide how to treat a global deadline.
            if self.deadline is not None and self.deadline_mode == "hard":
                latest_t = 0
                if self.time_of_engine:
                    latest_t = max(latest_t, max(self.time_of_engine.values()))
                if self.loc_time:
                    latest_t = max(latest_t, max(self.loc_time.values()))
                if latest_t > self.deadline + EPS_TIME:
                    self._record_violation(
                        0, "DEADLINE",
                        {"deadline_hours": self.deadline, "latest_hours": latest_t},
                        f"Deadline {hours_to_pretty(self.deadline)} missed (latest time {hours_to_pretty(latest_t)})"
                    )
                    goal_achieved = False
            # if self.deadline_mode == "horizon": no global violation here
        deliveries = { k: sorted(v) for k,v in self.arrivals.items() }
        return {
            "violations": list(self.violations),            # strings ok
            "violation_details": list(self.violation_details),  # already sanitized
            "goal_achieved": int(goal_achieved and len(self.violations) == 0),
        }

# ========== Goal & inventory evaluation ==========

    def _derive_goals_from_desc(self, desc: str) -> Dict[str, Dict[str,int]]:
        if not desc:
            return {}
        text = desc.lower()
        NUM_WORDS = {
        "one":1, "a":1, "an":1, "two":2, "three":3, "four":4, "five":5,
        "six":6, "seven":7, "eight":8, "nine":9, "ten":10
        }

        # What cargos can appear
        cargos = ["oranges", "bananas", "juice", "orange juice", "oj", "o.j.", "o.j"]
        # Normalize "orange juice" / "oj" to "juice"
        cargo_norm = lambda s: ("juice" if s in ("orange juice","oj","o.j.","o.j") else s)

        # Candidate destinations: any FACILITY node mentioned in the text
        loc_names = list(FACILITY.keys())
        found_locs = [loc for loc in loc_names if loc.lower() in text]

        # Heuristics to find qty (defaults to 1)
        qty = 1
        # phrases like "three boxcars", "2 tankers", "two", "2"
        m = re.search(r'\b(?:(one|two|three|four|five|six|seven|eight|nine|ten)|(\d+))\s+(?:boxcar|boxcars|tanker|tankers)\b', text)
        if m:
            if m.group(1): qty = NUM_WORDS[m.group(1)]
            elif m.group(2): qty = int(m.group(2))

        # Find cargo mention; prefer oranges/bananas before juice unless "juice" explicitly present
        found_cargo = None
        for c in cargos:
            if c in text:
                found_cargo = cargo_norm(c)
                # prefer exact words
                if re.search(rf'\b{re.escape(c)}\b', text):
                    found_cargo = cargo_norm(c)
                    break

        goals: Dict[str, Dict[str,int]] = {}
        if found_cargo and found_locs:
            # Pick the last location mentioned as the likely destination (e.g., "to Bath")
            # or any location preceded by "to|at|by|in"
            mloc = re.search(r'\b(?:to|at|by|in)\s+([A-Z][a-z]+)\b', desc)
            dest = None
            if mloc and mloc.group(1) in loc_names:
                dest = mloc.group(1)
            else:
                dest = found_locs[-1]

            goals.setdefault(found_cargo, {})[dest] = qty

        return goals

    def _inventory_by_location(self) -> Dict[str, Dict[str,int]]:
        """
        Returns counts by final location:
          inv[loc]["oranges"/"bananas"/"juice"] = count of loaded cars at that node,
        counting both yard cars and cars attached to any engine currently at that node.
        """
        inv: Dict[str, Dict[str,int]] = {loc: {} for loc in FACILITY.keys()}
        # yards
        for loc, cars in self.yards.items():
            for c in cars:
                if c["type"] == "boxcar" and c["payload"] in CARGO_BOX:
                    inv[loc][c["payload"]] = inv[loc].get(c["payload"], 0) + 1
                if c["type"] == "tanker" and c["payload"] == CARGO_LIQ:
                    inv[loc][CARGO_LIQ] = inv[loc].get(CARGO_LIQ, 0) + 1
        # attached to engines
        for eid, loc in self.loc_of_engine.items():
            for c in self.attached[eid]:
                if c["type"] == "boxcar" and c["payload"] in CARGO_BOX:
                    inv[loc][c["payload"]] = inv[loc].get(c["payload"], 0) + 1
                if c["type"] == "tanker" and c["payload"] == CARGO_LIQ:
                    inv[loc][CARGO_LIQ] = inv[loc].get(CARGO_LIQ, 0) + 1
        return inv

    def _check_goals(self) -> bool:
        # FIXED: If no goals are defined, this should be treated as an error, not success
        if not self.goals:
            self._record_violation(
                0, "GOAL", 
                {"error": "no_goals_defined"}, 
                "Goal: No delivery goals were defined for this problem"
            )
            return False
            
        inv = self._inventory_by_location()
        ok = True
        for cargo, dests in self.goals.items():
            cargo = norm_cargo(cargo)
            for dest, need in dests.items():
                have = inv.get(dest, {}).get(cargo, 0)
                if have < int(need):
                    msg = f"Goal: need {need} {cargo} at {dest}, have {have}"
                    # step index 0 to mean post-run/global
                    self._record_violation(0, "GOAL", {"cargo": cargo, "dest": dest, "need": need, "have": have}, msg)
                    ok = False
        return ok


        # ---------- New: read-only snapshot of current world ----------
    def snapshot(self) -> Dict[str, Any]:
        def inv_counts(arr):
            out = {"empty_boxcars":0,"empty_tankers":0,"oranges":0,"bananas":0,"juice":0}
            for c in arr:
                if c["type"]=="boxcar" and c["payload"]=="empty": out["empty_boxcars"] += 1
                if c["type"]=="tanker" and c["payload"]=="empty": out["empty_tankers"] += 1
                if c["type"]=="boxcar" and c["payload"]=="oranges": out["oranges"] += 1
                if c["type"]=="boxcar" and c["payload"]=="bananas": out["bananas"] += 1
                if c["type"]=="tanker" and c["payload"]=="juice": out["juice"] += 1
            return out

        yards = {loc: inv_counts(self.yards[loc]) for loc in self.yards}
        engines = {eid: {"loc": self.loc_of_engine[eid], "time_hours": self.time_of_engine[eid],
                         "attached":[dict(c) for c in self.attached[eid]]}
                   for eid in self.loc_of_engine}
        return {
            "goals": deepcopy(self.goals),
            "deadline": self.deadline,
            "earliest_arrivals": deepcopy(self.earliest_arrivals),
            "engines": engines,
            "yards": yards,
        }

    # ---------- New: check goals without adding violations ----------
    def goals_met(self) -> bool:
        # FIXED: If no goals are defined, goals are NOT met (this should be an error condition)
        if not self.goals:
            return False
            
        inv = self._inventory_by_location()
        for cargo, dests in self.goals.items():
            for dest, need in dests.items():
                if inv.get(dest, {}).get(cargo, 0) < int(need):
                    return False
        return True

    # ---------- New: internal state clone/restore for atomic tries ----------
    def _clone_state(self) -> Dict[str, Any]:
        return {
            "loc_of_engine": deepcopy(self.loc_of_engine),
            "time_of_engine": deepcopy(self.time_of_engine),
            "yards": deepcopy(self.yards),
            "attached": deepcopy(self.attached),
            "blocked": deepcopy(self.blocked),
            "arrivals": deepcopy(self.arrivals),
            "loc_time": deepcopy(self.loc_time),
            "must_depart_from": deepcopy(self.must_depart_from),
        }

    def _restore_state(self, s: Dict[str, Any]) -> None:
        self.loc_of_engine = s["loc_of_engine"]
        self.time_of_engine = s["time_of_engine"]
        self.yards = s["yards"]
        self.attached = s["attached"]
        self.blocked = s["blocked"]
        self.arrivals = s["arrivals"]
        self.loc_time = s["loc_time"]
        self.must_depart_from = s["must_depart_from"]

    # ---------- New: apply a single step atomically ----------
    def apply_step(self, step: Dict[str, Any], step_idx: int = 1) -> Optional[str]:
        save = self._clone_state()
        try:
            act = step.get("act")
            args = self._normalize_args(step.get("args", {}) or {})
            if act == "START":    self._exec_START(args)
            elif act == "ATTACH": self._exec_ATTACH(args)
            elif act == "DETACH": self._exec_DETACH(args)
            elif act == "LOAD":   self._exec_LOAD(args)
            elif act == "UNLOAD": self._exec_UNLOAD(args)
            elif act == "CONVERT":self._exec_CONVERT(args)
            elif act == "TRAVEL": self._exec_TRAVEL(args)
            elif act == "WAIT":   self._exec_WAIT(args)
            else:
                raise SimulationError(f"Unknown act {act}")
            return None  # success
        except SimulationError as e:
            # rollback
            self._restore_state(save)
            return f"s{step_idx}: {e}"
        

    # ---------- New: weighted shortest path over EDGE_HOURS ----------
    def shortest_path(self, a: str, b: str, start_hours: float = 0) -> Optional[List[str]]:
        forbid = _forbid_edges_from_constraints(self.constraints, start_hours)
        if a == b:
            return [a]

        # build adjacency, skipping forbidden edges
        adj: dict[str, list[tuple[str, int]]] = {}
        for (u, v), h in EDGE_HOURS.items():  # h = hours
            if tuple(sorted((u, v))) in forbid:
                continue
            adj.setdefault(u, []).append((v, h))
            adj.setdefault(v, []).append((u, h))

        import heapq
        pq = [(0, a, [a])]   # (cost_in_hours, node, path)
        seen: dict[str, int] = {}

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == b:
                return path
            if node in seen and seen[node] <= cost:
                continue
            seen[node] = cost
            for nxt, w in adj.get(node, []):
                heapq.heappush(pq, (cost + w, nxt, path + [nxt]))

        return None


    def travel_steps(self, engine: str, src: str, dst: str) -> List[Dict[str, Any]]:
        t0 = self.time_of_engine.get(engine, 0)
        path = self.shortest_path(src, dst, start_hours=t0) or []
        out = []
        for u,v in zip(path, path[1:]):
            out.append({"act":"TRAVEL","args":{"engine":engine,"from":u,"to":v}})
        return out

        # ---------- Deadline lower-bound helpers (network-only) ----------

    def _has_empty_boxcar(self, loc: str) -> bool:
        return any(c["type"] == "boxcar" and c["payload"] == "empty" for c in self.yards.get(loc, []))

    def _nearest_empty_boxcar_hours(self, from_loc: str) -> Optional[float]:
        """Shortest-path hours from from_loc to ANY yard with ≥1 empty boxcar."""
        best_h = None
        for loc, cars in self.yards.items():
            if any(c["type"] == "boxcar" and c["payload"] == "empty" for c in cars):
                path = self.shortest_path(from_loc, loc)
                if path and len(path) >= 2:
                    h = float(self.path_cost(path))
                    if best_h is None or h < best_h:
                        best_h = h
        return best_h

    def _lb_one_unit_hours(self, cargo: str, dest: str) -> Optional[float]:
        """
        A conservative (optimistic) network-only lower bound (hours) to deliver ONE unit
        of the given cargo to 'dest', ignoring capacity, single-track, and engine placement.
        - bananas: must LOAD at Avon
        - oranges: must LOAD at Corning
        - juice: (skip LB here; not needed for 2-C; return None so we don't assert infeasible)
        """
        cargo = norm_cargo(cargo)
        if cargo == "bananas":
            load_loc = "Avon"
        elif cargo == "oranges":
            load_loc = "Corning"
        else:
            # You can extend LB for 'juice' later (needs oranges+empty tankers+CONVERT at Elmira).
            return None

        # 1) Fetch empties to the load site if the yard has none (optimistic: one fetch round-trip)
        fetch_hours = 0.0
        if not self._has_empty_boxcar(load_loc):
            nearest_h = self._nearest_empty_boxcar_hours(load_loc)
            if nearest_h is None:
                return float("inf")  # no empties anywhere => impossible
            fetch_hours = 2 * nearest_h  # out-and-back

        # 2) LOAD (batch is 1h)
        load_hours = float(DUR["LOAD"])

        # 3) Travel load_site -> dest (shortest path)
        path = self.shortest_path(load_loc, dest)
        if not path or len(path) < 2:
            return float("inf")  # unreachable
        travel_hours = float(self.path_cost(path))

        return fetch_hours + load_hours + travel_hours



# ========== Public API ==========
def check_plan(plan: Dict[str, Any], problem_desc: str, goals: Dict[str, Dict[str,int]], constraints: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the plan and return:
      {"violations": [...], "goal_achieved": 0/1}
    """
    goals       = goals or {}
    constraints = constraints or {}
    overrides   = overrides or {}

    # Only fallback-parse when caller didn't pass goals
    if not goals and problem_desc:
        parsed = parse_problem_description(problem_desc) or {}
        goals       = parsed.get("deliver", {}) or {}
        constraints = {**parsed.get("constraints", {}), **constraints}
        overrides   = {**parsed.get("overrides", {}),   **overrides}

    chk = Checker(problem_desc=problem_desc,
                  goals=goals,
                  constraints=constraints,
                  overrides=overrides)

    if getattr(chk, "debug", False):
        print(f"[CHECKER] goals in use = {chk.goals}")
    return chk.run(plan)
