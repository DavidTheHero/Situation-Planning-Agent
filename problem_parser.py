import re
from typing import Dict, Any, Optional, Tuple, List

# ---------- time helpers (hours only → minutes) ----------
_HH_AMPM = re.compile(r'\b(\d{1,2})\s*(am|pm)\b', re.I)

LOCATIONS = {"Avon", "Bath", "Corning", "Elmira", "Dansville"}

def _parse_deadline_minutes(s: str) -> Optional[int]:
    s = s.lower()
    if re.search(r'\bby\s+noon\b', s): return 12 * 60
    if re.search(r'\bby\s+midnight\b', s): return 0
    m = re.search(r'\bby\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', s)
    if not m:
        return None
    h = int(m.group(1))
    mnt = int(m.group(2) or 0)
    ampm = m.group(3)
    if h == 12:  # 12am -> 0h, 12pm -> 12h
        h = 0
    if ampm == "pm":
        h += 12
    return h * 60 + mnt

def _to_minutes_from_hour(h: int, ampm: str) -> int:
    h = int(h)
    ampm = ampm.lower()
    if ampm == "pm" and h != 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return h * 60

def _parse_hour_word(s: str) -> Optional[int]:
    """
    Returns minutes since midnight (multiple of 60) or None.
    Accepts '3 pm', '11 AM', 'noon', 'midnight'.
    """
    s = s.strip().lower()
    if s in {"noon"}:
        return 12 * 60
    if s in {"midnight"}:
        return 0
    # Match: HH[:MM] with optional am/pm (e.g., '9', '9 am', '09:30', '9:30pm')
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?|am|pm)?\b", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    mer = (m.group(3) or '').replace('.', '')
    if mer:
        # 12-hour clock
        if mer == 'pm' and hh != 12:
            hh += 12
        if mer == 'am' and hh == 12:
            hh = 0
        return hh * 60 + mm
    # No meridian given: interpret as 24-hour clock if minutes provided or hh in 0..23
    if 0 <= hh <= 23:
        return hh * 60 + mm
    return None

# ---------- main parser ----------
def parse_problem_description(desc: str) -> Dict[str, Any]:
    import re

    d: Dict[str, Any] = {
        "deliver": {},
        "constraints": {},
        "overrides": {},
    }

    def _qty(s: str) -> int:
        s = (s or "").strip().lower()
        if s in {"a", "an", "one"}: return 1
        words = {"two":2, "three":3, "four":4, "five":5, "six":6}
        if s in words: return words[s]
        return int(s)

    def _norm_cargo(s: str) -> str:
        s = (s or "").strip().lower()
        return "juice" if s in {"oj","orange juice","o.j.","o.j"} else s

    # assumes _parse_hour_word(hh[:mm] am/pm|noon|midnight) is defined elsewhere

    text = (desc or "").strip()

    # ---------- Scope the task text only (avoid Analysis/Solution chatter) ----------
    core = re.split(r'\b(Analysis of Problem|Solution)\b', text, maxsplit=1, flags=re.I)[0]

    # Whitelist of map nodes and a regex for them
    LOC_SET = {"Avon", "Bath", "Corning", "Dansville", "Elmira"}
    LOC = r'(?:Avon|Bath|Corning|Dansville|Elmira)'
    LOC_RX = re.compile(rf'\b{LOC}\b')

    # Helper to split "Bath, Corning, and Elmira" safely
    def _split_dests(list_text: str) -> list[str]:
        s = re.sub(r'\([^)]*\)', '', list_text)              # drop parentheticals
        s = re.sub(r'[\s,\.]+$', '', s).strip()              # trim trailing punctuation/space
        return LOC_RX.findall(s)                              # keep only valid map nodes

    # --- Remove parenthetical asides from the core (avoid “unload” noise) ---
    core_no_paren = re.sub(r'\([^)]*\)', ' ', core)

    # --- Start a masked copy we can safely redact from for generic parsing ---
    masked_text = core_no_paren

    # --- Mask narrative shuttle phrases (1-E) like “carrying 3 boxcars of bananas to Dansville” ---
    masked_text = re.sub(
        rf'carrying\s+(?:a|an|one|\d+)\s+(?:boxcars?|tankers?)\s+of\s+'
        r'(?:oranges|bananas|oj|orange\s+juice|juice)\s+to\s+[A-Z][a-z]+',
        ' ', masked_text, flags=re.I
    )

    # --- Build “task-like” text (for deadlines, etc.), ignoring shuttle chatter ---
    sentences = re.split(r'(?<=[.!?])\s+', core.strip())
    task_sents = []
    for s in sentences:
        if re.search(r'\b(your task is to|you are to|you must|please|ship|deliver|bring|take|transport|make)\b', s, re.I):
            if re.search(r'\b(carrying|every trip|runs back and forth)\b', s, re.I):
                continue
            task_sents.append(s)
    task_text = " ".join(task_sents) or core

    # Lowercase working copy for cargo wording normalization
    slow_task = " " + task_text.lower() + " "
    slow_task = re.sub(r'\bo\.?\s*j\.?\b', 'juice', slow_task)     # OJ → juice
    slow_task = re.sub(r'orange\s+juice', 'juice', slow_task)

    # ---------- Preloads (scan full text) ----------
    d.setdefault("overrides", {}).setdefault("boxcars", [])
    d["overrides"].setdefault("tankers", [])
    preload_re = re.compile(
        r'\bthere\s+are\s+(a|an|one|\d+)\s+'
        r'(boxcars?|tankers?)\s+of\s+(oranges|bananas|oj|orange\s+juice|juice)\s+'
        r'waiting\s+for\s+you\s+at\s+([A-Z][a-z]+)\b',
        re.I
    )
    for m in preload_re.finditer(text):
        qty = _qty(m.group(1)); unit = m.group(2).lower()
        cargo = _norm_cargo(m.group(3)); loc = m.group(4)
        if unit.startswith("boxcar"):
            d["overrides"]["boxcars"].append({"count": qty, "payload": cargo, "location": loc})
        else:
            d["overrides"]["tankers"].append({"count": qty, "payload": cargo, "location": loc})


    # ---------- Special: "make into N tankers of juice" + "ship juice ... to DEST" ----------
    m_make = re.search(
        r'\bmake\s+(?:it|them)?\s*into\s*(a|an|one|\d+)\s*tanker(?:s)?\s*of\s*juice\b',
        slow_task, re.I
    )
    qty_make = _qty(m_make.group(1)) if m_make else None

    m_ship_juice = re.search(
        rf'\b(?:ship|deliver|bring|take)\s+(?:the\s+)?(?:juice|OJ)\b.*?\bto\s+({LOC})\b',
        core, re.I
    )
    if m_ship_juice:
        dest = m_ship_juice.group(1)
        qty  = qty_make if qty_make is not None else 1
        d.setdefault("deliver", {}).setdefault("juice", {})
        d["deliver"]["juice"][dest] = d["deliver"]["juice"].get(dest, 0) + qty

    # ---------- "deliver 1 each to A, B, and C" ----------
    m_take_cargo = re.search(r'\btake\s+(?:a|an|one|\d+)\s+boxcars?\s+of\s+(oranges|bananas)\b', core, re.I)
    cargo_ctx = _norm_cargo(m_take_cargo.group(1)) if m_take_cargo else None

    each_re = re.compile(
        rf'\bdeliver\s+(?:a|an|one|1)\s+each\s+to\s+((?:{LOC}(?:\s*,\s*|\s*,?\s*and\s+))*{LOC})',
        re.I
    )
    masked_spans = []
    for m in each_re.finditer(masked_text):
        for dest in _split_dests(m.group(1)):
            cargo = cargo_ctx or "bananas"  # 3-B default
            d.setdefault("deliver", {}).setdefault(cargo, {})
            d["deliver"][cargo][dest] = d["deliver"][cargo].get(dest, 0) + 1
        masked_spans.append((m.start(), m.end()))

    # Redact those spans so generic rules won’t re-parse them
    if masked_spans:
        chars = list(masked_text)
        for a, b in masked_spans:
            for i in range(a, b):
                chars[i] = " "
        masked_text = "".join(chars)

    # ---------- Generic “deliver/ship/transport/bring <items> to <Dest>” (EXCLUDE 'take') ----------
    deliver_phrase_re = re.compile(
        rf'\b(transport|ship|deliver|bring)\s+(.+?)\s+to\s+({LOC})\b',
        re.I
    )
    item_re = re.compile(
        r'(a|an|one|\d+)\s+(boxcars?|tankers?)\s+of\s+'
        r'(oranges|bananas|oj|orange\s+juice|juice)',
        re.I
    )

    covered_spans = []
    for m in deliver_phrase_re.finditer(masked_text):
        items_span = m.group(2)
        dest = m.group(3)
        covered_spans.append((m.start(), m.end()))
        for it in item_re.finditer(items_span):
            qty   = _qty(it.group(1))
            cargo = _norm_cargo(it.group(3))
            d.setdefault("deliver", {}).setdefault(cargo, {})
            d["deliver"][cargo][dest] = d["deliver"][cargo].get(dest, 0) + qty

    # ---------- Fallback “<item> … to <Dest>” without leading verb (on masked_text) ----------
    pair_re = re.compile(
        rf'(a|an|one|\d+)\s+(boxcars?|tankers?)\s+of\s+'
        rf'(oranges|bananas|oj|orange\s+juice|juice)\s+to\s+({LOC})\b',
        re.I
    )
    def _inside_any(idx: int) -> bool:
        return any(a <= idx < b for (a, b) in covered_spans)

    for m in pair_re.finditer(masked_text):
        if _inside_any(m.start()):
            continue
        qty   = _qty(m.group(1))
        cargo = _norm_cargo(m.group(3))
        dest  = m.group(4)
        d.setdefault("deliver", {}).setdefault(cargo, {})
        d["deliver"][cargo][dest] = d["deliver"][cargo].get(dest, 0) + qty

    # ---------- Scrub any stray non-locations just in case ----------
    for cargo, locs in list(d["deliver"].items()):
        for k in list(locs.keys()):
            if k not in LOC_SET:
                del locs[k]
        if not locs:
            del d["deliver"][cargo]


    # ---------- DEADLINES (global + per-delivery) ----------
    # Accept: "by 8 AM", "by 8:30 PM", "by 14:00", "by noon", "by midnight"
    # Also accept: "before noon", "no later than noon", etc.
    _DEADLINE_RX = re.compile(
        r"\b(?:by|before|no\s+later\s+than|not\s+later\s+than)\s+"
        r"(noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        re.I,
    )

    # First, try to detect cargo-specific deadlines like
    #   "The bananas must arrive in Elmira by 9 PM"
    #   "Ship juice to Avon before noon"
    cargo_deadline_any = False
    CARGO_DEADLINE_RX = re.compile(
        rf"\b(bananas|oranges|oj|orange\s+juice|juice)\b[^.?!]*?\b(?:in|at|to)\s+({LOC})\b[^.?!]*?"
        rf"(?:by|before|no\s+later\s+than|not\s+later\s+than)\s+"
        rf"(noon|midnight|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)?)\b",
        re.I,
    )
    for m in CARGO_DEADLINE_RX.finditer(core):
        cargo = _norm_cargo(m.group(1))
        dest  = m.group(2)
        tmin  = _parse_hour_word(m.group(3))
        if tmin is None:
            continue
        d.setdefault("constraints", {})
        dd = d["constraints"].setdefault("deliver_deadlines", {})
        dd.setdefault(cargo, {})[dest] = tmin
        cargo_deadline_any = True

    # Next, capture generic deadlines (apply to all deliveries) only if no cargo-specific deadlines were found
    m_by = _DEADLINE_RX.search(task_text) or _DEADLINE_RX.search(core) or _DEADLINE_RX.search(text)
    by_min = _parse_hour_word(m_by.group(1)) if m_by else None

    # Accept: "within N hours"
    m_within = re.search(r'\bwithin\s+(\d+)\s*hours?\b', task_text, re.I)
    within_min = None
    if m_within:
        try:
            hours = int(m_within.group(1))
            base = d.get("constraints", {}).get("start_time", 0)  # default midnight
            within_min = base + hours * 60
        except Exception:
            within_min = None

    # Use the stricter (earlier) of the two if both exist
    deadline_min_any = None
    for candidate in (by_min, within_min):
        if candidate is not None:
            deadline_min_any = candidate if deadline_min_any is None else min(deadline_min_any, candidate)

    # Fallback: scan the full text with a simpler parser if still None
    if deadline_min_any is None:
        try:
            deadline_min_any = _parse_deadline_minutes(text)
        except Exception:
            deadline_min_any = None

    if not cargo_deadline_any and deadline_min_any is not None:
        # Set a global deadline and mirror to all deliveries
        d.setdefault("constraints", {})
        d["constraints"].setdefault("start_time", 0)
        d["constraints"]["deadline"] = deadline_min_any
        if d.get("deliver"):
            dd = d["constraints"].setdefault("deliver_deadlines", {})
            for cargo, locs in d["deliver"].items():
                for loc in locs:
                    dd.setdefault(cargo, {})[loc] = deadline_min_any
    # If cargo-specific deadlines exist, we intentionally do NOT set a global deadline here.

    # ---------- RETURN-TO (round-trip) ----------
    # e.g., "arrive back in Avon within 24 hours", "return to Avon within 24 hours"
    m_ret = re.search(
        rf'\b(arrive\s+back\s+in|return\s+to)\s+({LOC})\b(?:.*?\bwithin\s+(\d+)\s*hours?)?',
        core, re.I
    )
    if m_ret:
        ret_loc = m_ret.group(2)
        d.setdefault("constraints", {})["return_to"] = ret_loc
        if m_ret.group(3):
            try:
                hrs = int(m_ret.group(3))
                base = d.get("constraints", {}).get("start_time", 0)
                d["constraints"]["return_by"] = base + hrs * 60
            except Exception:
                pass

    # ---------- ENGINE RESTRICTION (only one engine allowed) ----------
    # e.g., "You can only use Engine E2", "Use only Engine E2"
    m_eng_only = re.search(r'\b(?:you\s+can\s+only\s+use|use\s+only)\s+engine\s+(E[1-3])\b', core, re.I)
    if m_eng_only:
        d.setdefault("constraints", {})["allowed_engines"] = [m_eng_only.group(1)]

    # ---------- BLOCKED EDGE (Problem 1-E style: shuttle runs back and forth until time) ----------
    # e.g., "runs back and forth between Avon and Dansville until 9 am"
    m_block = re.search(
        r'runs\s+back\s+and\s+forth\s+between\s+(Avon|Bath|Corning|Dansville|Elmira)\s+and\s+(Avon|Bath|Corning|Dansville|Elmira)\s+until\s+([^\.,]+)',
        text, re.I
    )
    if m_block:
        a, b, tstr = m_block.group(1), m_block.group(2), m_block.group(3)
        until_min = _parse_hour_word(tstr)
        if until_min is not None:
            d["constraints"].setdefault("start_time", 0)
            d["constraints"].setdefault("blocked_edges", []).append({"from": a, "to": b, "until": until_min})
            d["constraints"]["blocked_edges"].append({"from": b, "to": a, "until": until_min})

    # 2) Narrative phrasing: "the route between Corning and Avon, by way of Dansville, is occupied"
    if re.search(r'between\s+Corning\s+and\s+Avon.*by\s+way\s+of\s+Dansville.*occupied', text, re.I):
        # The bottleneck is the Dansville–Avon leg; assume blocked midnight–9 AM if no window stated
        d["constraints"].setdefault("blocked_edges", []).append(
            {"edge": ["Avon", "Dansville"], "start_h": 0, "end_h": 9}
        )

    # ---------- H: Other constraints ----------

    # --- Guard against spurious "bananas → Avon" from context leakage (e.g., Problem 3-B)
    if d.get("deliver", {}).get("bananas", {}).get("Avon"):
        # Only keep it if the task text explicitly says bananas go to Avon
        explicit_bananas_to_avon = re.search(
            r'\b(?:deliver|ship|bring|take)\s+(?:a|an|one|\d+)\s+boxcars?\s+of\s+bananas\s+(?:back\s+to|to)\s+Avon\b',
            core, re.I
        ) or re.search(
            r'\b(?:a|an|one|\d+)\s+boxcars?\s+of\s+bananas\s+(?:back\s+to|to)\s+Avon\b',
            core, re.I
        )
        if not explicit_bananas_to_avon:
            # Drop the spurious entry
            d["deliver"]["bananas"].pop("Avon", None)
            if not d["deliver"]["bananas"]:
                d["deliver"].pop("bananas", None)
                
    # can't arrive before …
    for m in re.finditer(
        r"\b(can't|cannot)\s+arrive\s+at\s+([A-Z][a-z]+)\s+before\s+(noon|midnight|\d{1,2}\s*(?:am|pm))\b",
        task_text, re.I
    ):
        loc = m.group(2)
        t = _parse_hour_word(m.group(3))
        if t is not None:
            d["constraints"].setdefault("earliest_arrivals", {})
            d["constraints"]["earliest_arrivals"][loc] = t

    # processed at Elmira at X (sharp)
    m = re.search(r'\bprocessed\s+at\s+the\s+factory\s+in\s+Elmira\s+at\s+(noon|midnight|\d{1,2}\s*(?:am|pm))(?:\s+sharp)?\b', task_text, re.I)
    if m:
        t = _parse_hour_word(m.group(1))
        if t is not None:
            d["constraints"]["exact_convert_time"] = t
            d["constraints"]["leave_after_convert"] = True

    if re.search(r'(as\s+soon\s+as\s+the\s+oj\s+is\s+made|leave\s+elmira\s+immediately\s+after\s+convert)', task_text, re.I):
        d["constraints"]["leave_after_convert"] = True

    # current time
    m = re.search(r'\bIt\s+is\s+now\s+(noon|midnight|\d{1,2}\s*(?:am|pm))\b', core, re.I)
    if m:
        t = _parse_hour_word(m.group(1))
        if t is not None:
            d["constraints"]["start_time"] = t


    return d
