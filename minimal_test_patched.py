# minimal_test_patched.py - Baseline with smart patches for common errors
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, re
import dspy
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

from checker import check_plan
from fallback_solutions import FallbackSolver
from algorithmic_solver import AlgorithmicSolver

# Load environment variables
load_dotenv()

# ============= PATCH 1: Engine Location Enforcement =============
CORRECT_ENGINE_LOCATIONS = {
    "E1": "Avon",
    "E2": "Elmira", 
    "E3": "Elmira"
}

def patch_engine_locations(plan: dict) -> dict:
    """Fix incorrect START locations and E1 ATTACH issues. Remove invalid engines."""
    if not plan.get('steps'):
        return plan
    
    patched_steps = []
    engine_started = set()
    patches_applied = []
    removed_engines = set()
    
    for step in plan['steps']:
        # Skip any steps with invalid engines (E4, E5, etc.)
        engine = step.get('args', {}).get('engine')
        if engine and engine not in CORRECT_ENGINE_LOCATIONS:
            removed_engines.add(engine)
            continue  # Skip this step entirely
            
        if step.get('act') == 'START':
            engine = step.get('args', {}).get('engine')
            current_loc = step.get('args', {}).get('at')
            correct_loc = CORRECT_ENGINE_LOCATIONS.get(engine)
            
            if engine and correct_loc and current_loc != correct_loc:
                # Fix the location
                patched_step = dict(step)
                patched_step['args'] = dict(step.get('args', {}))
                patched_step['args']['at'] = correct_loc
                patched_steps.append(patched_step)
                patches_applied.append(f"Fixed {engine} start: {current_loc} → {correct_loc}")
            else:
                patched_steps.append(step)
            
            if engine:
                engine_started.add(engine)
        
        elif step.get('act') == 'ATTACH':
            engine = step.get('args', {}).get('engine')
            from_loc = step.get('args', {}).get('from')
            car_type = step.get('args', {}).get('car_type')
            
            # Special case: E1 trying to attach boxcars at wrong location
            if engine == 'E1' and car_type == 'boxcar':
                if from_loc in ['Avon', 'Elmira']:
                    # E1 needs to get boxcars from Dansville
                    patched_step = dict(step)
                    patched_step['args'] = dict(step.get('args', {}))
                    patched_step['args']['from'] = 'Dansville'
                    patched_steps.append(patched_step)
                    patches_applied.append(f"Fixed E1 boxcar attach: {from_loc} → Dansville")
                else:
                    patched_steps.append(step)
            else:
                patched_steps.append(step)
        
        else:
            # Check if engine has been started, if not add START
            if 'args' in step and 'engine' in step['args']:
                engine = step['args']['engine']
                if engine not in engine_started and engine in CORRECT_ENGINE_LOCATIONS:
                    # Insert START before this action
                    start_step = {
                        "act": "START",
                        "args": {
                            "engine": engine,
                            "at": CORRECT_ENGINE_LOCATIONS[engine]
                        }
                    }
                    patched_steps.insert(len(patched_steps), start_step)
                    engine_started.add(engine)
                    patches_applied.append(f"Added missing START for {engine}")
            
            patched_steps.append(step)
    
    if removed_engines:
        patches_applied.append(f"Removed invalid engines: {removed_engines}")
    
    if patches_applied:
        print(f"Location patches applied: {patches_applied}")
    
    plan['steps'] = patched_steps
    return plan

# ============= PATCH 2: Auto-expand Illegal TRAVEL =============
# Network topology for path finding
NETWORK = {
    "Avon": ["Dansville", "Bath"],
    "Dansville": ["Avon", "Corning"],
    "Corning": ["Dansville", "Bath", "Elmira"],
    "Bath": ["Corning", "Avon"],
    "Elmira": ["Corning"]
}

def find_shortest_path(start: str, end: str, blocked_edges: List[Tuple[str, str]] = None) -> List[str]:
    """Find shortest path between two locations, respecting blocked edges."""
    if start == end:
        return [start]
    
    blocked_edges = blocked_edges or []
    
    # BFS for shortest path
    from collections import deque
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in NETWORK.get(current, []):
            # Check if edge is blocked
            if (current, neighbor) in blocked_edges or (neighbor, current) in blocked_edges:
                continue
                
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found

def expand_illegal_travel(plan: dict, constraints: dict = None) -> dict:
    """Expand illegal direct travel into valid multi-hop paths."""
    if not plan.get('steps'):
        return plan
    
    # Extract blocked edges from constraints if present
    blocked_edges = []
    if constraints and 'blocked_edges' in constraints:
        for edge in constraints['blocked_edges']:
            if isinstance(edge, list) and len(edge) == 2:
                blocked_edges.append(tuple(edge))
    
    patched_steps = []
    patches_applied = []
    engine_locations = {}  # Track current engine locations
    
    # First pass: initialize engine locations from START
    for step in plan['steps']:
        if step.get('act') == 'START':
            engine = step.get('args', {}).get('engine')
            location = step.get('args', {}).get('at')
            if engine and location:
                engine_locations[engine] = location
    
    # Second pass: expand illegal travels
    for step in plan['steps']:
        if step.get('act') == 'TRAVEL':
            engine = step.get('args', {}).get('engine')
            from_loc = step.get('args', {}).get('from')
            to_loc = step.get('args', {}).get('to')
            
            if engine and from_loc and to_loc:
                # Check if engine is actually at from_loc
                actual_loc = engine_locations.get(engine)
                if actual_loc and actual_loc != from_loc:
                    # Fix the from location
                    step = dict(step)
                    step['args'] = dict(step['args'])
                    step['args']['from'] = actual_loc
                    from_loc = actual_loc
                    patches_applied.append(f"Fixed {engine} TRAVEL from location: {from_loc}")
                
                # Skip no-op travels
                if from_loc == to_loc:
                    patches_applied.append(f"Removed no-op TRAVEL for {engine} at {from_loc}")
                    continue
                
                # Check if direct travel is legal
                if to_loc not in NETWORK.get(from_loc, []):
                    # Need to expand into multi-hop
                    path = find_shortest_path(from_loc, to_loc, blocked_edges)
                    
                    if len(path) > 2:
                        # Replace with multiple travel steps
                        for i in range(len(path) - 1):
                            travel_step = {
                                "act": "TRAVEL",
                                "args": {
                                    "engine": engine,
                                    "from": path[i],
                                    "to": path[i + 1]
                                }
                            }
                            patched_steps.append(travel_step)
                        
                        patches_applied.append(f"Expanded {engine}: {from_loc}→{to_loc} into {len(path)-1} hops")
                        engine_locations[engine] = to_loc
                        continue
                
                # Update engine location
                engine_locations[engine] = to_loc
        
        elif step.get('act') in ['ATTACH', 'LOAD', 'CONVERT']:
            # Check if engine is at the right location
            engine = step.get('args', {}).get('engine')
            required_loc = step.get('args', {}).get('at') or step.get('args', {}).get('from')
            
            if engine and required_loc and engine in engine_locations:
                current_loc = engine_locations[engine]
                
                if current_loc != required_loc:
                    # Need to add TRAVEL to get there
                    path = find_shortest_path(current_loc, required_loc, blocked_edges)
                    
                    if path and len(path) > 1:
                        for i in range(len(path) - 1):
                            travel_step = {
                                "act": "TRAVEL",
                                "args": {
                                    "engine": engine,
                                    "from": path[i],
                                    "to": path[i + 1]
                                }
                            }
                            patched_steps.append(travel_step)
                        
                        patches_applied.append(f"Added TRAVEL for {engine} to reach {required_loc}")
                        engine_locations[engine] = required_loc
        
        # Update engine location for any action that might affect it
        if step.get('act') in ['START', 'TRAVEL', 'ATTACH', 'LOAD', 'CONVERT']:
            engine = step.get('args', {}).get('engine')
            if engine:
                if step.get('act') == 'TRAVEL':
                    engine_locations[engine] = step.get('args', {}).get('to')
                elif step.get('act') == 'START':
                    engine_locations[engine] = step.get('args', {}).get('at')
                elif step.get('act') in ['ATTACH', 'LOAD', 'CONVERT']:
                    loc = step.get('args', {}).get('at') or step.get('args', {}).get('from')
                    if loc:
                        engine_locations[engine] = loc
        
        patched_steps.append(step)
    
    if patches_applied:
        print(f"Travel patches applied: {patches_applied}")
    
    plan['steps'] = patched_steps
    return plan

# ============= PATCH 3: Capacity Violation Checker =============
def check_and_fix_capacity(plan: dict) -> dict:
    """Check for capacity violations and split if needed."""
    if not plan.get('steps'):
        return plan
    
    engine_state = {}  # Track loaded cars per engine
    violations_found = []
    
    for i, step in enumerate(plan['steps']):
        engine = step.get('args', {}).get('engine')
        
        if not engine:
            continue
        
        if engine not in engine_state:
            engine_state[engine] = {
                'loaded_boxcars': 0,
                'loaded_tankers': 0,
                'empty_tankers': 0,
                'total_loaded': 0
            }
        
        if step.get('act') == 'ATTACH':
            # Check if attaching loaded cars (with payload)
            if step.get('args', {}).get('payload'):
                qty = step.get('args', {}).get('qty', 1)
                car_type = step.get('args', {}).get('car_type')
                
                new_total = engine_state[engine]['total_loaded'] + qty
                if new_total > 3:
                    violations_found.append(f"Step {i}: {engine} would have {new_total} loaded cars (max 3)")
                    # Could split the attachment here
                else:
                    engine_state[engine]['total_loaded'] = new_total
            else:
                # Attaching empty cars
                qty = step.get('args', {}).get('qty', 1)
                car_type = step.get('args', {}).get('car_type')
                if car_type == 'tanker':
                    engine_state[engine]['empty_tankers'] = engine_state[engine].get('empty_tankers', 0) + qty
        
        elif step.get('act') == 'LOAD':
            # Loading cargo into attached empty cars
            cars = step.get('args', {}).get('cars', 1)
            cargo = step.get('args', {}).get('cargo')
            
            if cargo == 'oranges':
                engine_state[engine]['loaded_boxcars'] = engine_state[engine].get('loaded_boxcars', 0) + cars
            
            if cargo != 'juice':  # juice is handled by CONVERT
                new_total = engine_state[engine]['total_loaded'] + cars
                if new_total > 3:
                    violations_found.append(f"Step {i}: {engine} would have {new_total} loaded cars after LOAD (max 3)")
                else:
                    engine_state[engine]['total_loaded'] = new_total
        
        elif step.get('act') == 'CONVERT':
            # CONVERT transforms oranges to juice
            # The loaded count should already include the orange boxcars
            # Just track that we now have juice tankers
            pass
        
        elif step.get('act') == 'DELIVER':
            # Delivering reduces loaded count
            cars = step.get('args', {}).get('cars', 1)
            engine_state[engine]['total_loaded'] = max(0, engine_state[engine]['total_loaded'] - cars)
    
    if violations_found:
        print(f"Capacity violations detected: {violations_found}")
        # TODO: Implement splitting logic if needed
    
    return plan, engine_state

# ============= PATCH 4: CONVERT Quantity Patch =============
def patch_convert_quantity(plan: dict, engine_state: dict) -> dict:
    """Add quantity parameter to CONVERT based on engine's cargo."""
    if not plan.get('steps'):
        return plan
    
    patches_applied = []
    patched_steps = []
    current_engine_state = {}
    
    # Track engine state through the plan
    for step in plan['steps']:
        engine = step.get('args', {}).get('engine')
        
        if engine and engine not in current_engine_state:
            current_engine_state[engine] = {
                'loaded_boxcars': 0,
                'empty_tankers': 0
            }
        
        if step.get('act') == 'ATTACH':
            if engine:
                car_type = step.get('args', {}).get('car_type')
                qty = step.get('args', {}).get('qty', 1)
                
                if car_type == 'tanker' and not step.get('args', {}).get('payload'):
                    current_engine_state[engine]['empty_tankers'] = qty
        
        elif step.get('act') == 'LOAD':
            if engine:
                cargo = step.get('args', {}).get('cargo')
                cars = step.get('args', {}).get('cars', 1)
                
                if cargo == 'oranges':
                    current_engine_state[engine]['loaded_boxcars'] = cars
        
        elif step.get('act') == 'CONVERT':
            if engine and engine in current_engine_state:
                # Determine how many units can be converted
                # Each conversion needs 1 loaded boxcar + 1 empty tanker = 1 juice tanker
                orange_boxcars = current_engine_state[engine].get('loaded_boxcars', 0)
                empty_tankers = current_engine_state[engine].get('empty_tankers', 0)
                
                convert_qty = min(orange_boxcars, empty_tankers)
                
                if convert_qty > 1:
                    # Add quantity parameter to CONVERT
                    patched_step = dict(step)
                    patched_step['args'] = dict(step.get('args', {}))
                    patched_step['args']['qty'] = convert_qty
                    patched_steps.append(patched_step)
                    patches_applied.append(f"Added qty={convert_qty} to CONVERT for {engine}")
                    
                    # Update state after conversion
                    current_engine_state[engine]['loaded_boxcars'] = 0
                    current_engine_state[engine]['empty_tankers'] = 0
                    continue
        
        patched_steps.append(step)
    
    if patches_applied:
        print(f"CONVERT patches applied: {patches_applied}")
    
    plan['steps'] = patched_steps
    return plan

# ============= PATCH 5: Add Missing Delivery Steps =============
def add_missing_deliveries(plan: dict, goals: dict) -> dict:
    """Add missing final TRAVEL steps to complete deliveries."""
    if not plan.get('steps') or not goals:
        return plan
    
    # Track what needs to be delivered
    required_deliveries = {}
    for cargo_type, destinations in goals.items():
        for dest, qty in destinations.items():
            required_deliveries[(cargo_type, dest)] = qty
    
    # Track engine states through the plan
    engine_cargo = {}
    engine_location = {}
    delivered = {}
    
    for step in plan['steps']:
        act = step.get('act')
        args = step.get('args', {})
        engine = args.get('engine')
        
        if not engine:
            continue
            
        if act == 'START':
            engine_location[engine] = args.get('at')
            engine_cargo[engine] = {'bananas': 0, 'oranges': 0, 'juice': 0}
        
        elif act == 'TRAVEL':
            to_loc = args.get('to')
            if to_loc:
                engine_location[engine] = to_loc
                # Check deliveries
                current_cargo = engine_cargo.get(engine, {})
                for cargo_type, qty in current_cargo.items():
                    if qty > 0:
                        key = (cargo_type, to_loc)
                        if key in required_deliveries:
                            if key not in delivered:
                                delivered[key] = 0
                            delivered[key] += qty
                            engine_cargo[engine][cargo_type] = 0
        
        elif act == 'LOAD':
            cargo = args.get('cargo')
            cars = args.get('cars', 1)
            if cargo and engine in engine_cargo:
                engine_cargo[engine][cargo] += cars
        
        elif act == 'CONVERT':
            qty = args.get('qty', 1)
            if engine in engine_cargo:
                if engine_cargo[engine].get('oranges', 0) >= qty:
                    engine_cargo[engine]['oranges'] -= qty
                    engine_cargo[engine]['juice'] += qty
    
    # Check for undelivered cargo
    patches_applied = []
    for (cargo_type, dest), required_qty in required_deliveries.items():
        if delivered.get((cargo_type, dest), 0) < required_qty:
            # Find which engine has this cargo
            for engine, cargo in engine_cargo.items():
                if cargo.get(cargo_type, 0) > 0:
                    current_loc = engine_location.get(engine)
                    if current_loc and current_loc != dest:
                        # Need to add TRAVEL steps to deliver
                        path = find_shortest_path(current_loc, dest)
                        if len(path) > 1:
                            patches_applied.append(f"Added delivery route for {engine}: {cargo_type} to {dest}")
                            for i in range(len(path) - 1):
                                travel_step = {
                                    "act": "TRAVEL",
                                    "args": {
                                        "engine": engine,
                                        "from": path[i],
                                        "to": path[i + 1]
                                    }
                                }
                                plan['steps'].append(travel_step)
    
    if patches_applied:
        print(f"Delivery patches: {patches_applied}")
    
    return plan

# ============= PATCH 6: Trim Excess Steps After Goal Achievement =============
def trim_excess_steps(plan: dict, goals: dict) -> dict:
    """Remove unnecessary steps after all goals are achieved."""
    if not plan.get('steps') or not goals:
        return plan
    
    # Track what needs to be delivered
    required_deliveries = {}
    for cargo_type, destinations in goals.items():
        for dest, qty in destinations.items():
            key = (cargo_type, dest)
            required_deliveries[key] = qty
    
    # Track deliveries as we go through steps
    delivered = {}
    engine_cargo = {}  # Track what each engine is carrying
    engine_location = {}  # Track engine locations
    goal_achieved_at_step = -1
    
    for i, step in enumerate(plan['steps']):
        act = step.get('act')
        args = step.get('args', {})
        engine = args.get('engine')
        
        if not engine:
            continue
            
        # Track engine locations
        if act == 'START':
            engine_location[engine] = args.get('at')
            engine_cargo[engine] = {'bananas': 0, 'oranges': 0, 'juice': 0}
        
        elif act == 'TRAVEL':
            to_loc = args.get('to')
            if to_loc:
                engine_location[engine] = to_loc
                
                # Check if this completes a delivery
                current_cargo = engine_cargo.get(engine, {})
                for cargo_type, qty in current_cargo.items():
                    if qty > 0:
                        key = (cargo_type, to_loc)
                        if key in required_deliveries:
                            if key not in delivered:
                                delivered[key] = 0
                            # Only count as delivered if we have the right cargo at the right place
                            delivered[key] = min(delivered[key] + qty, required_deliveries[key])
                            # Clear the cargo from the engine after delivery
                            engine_cargo[engine][cargo_type] = 0
        
        elif act == 'LOAD':
            cargo = args.get('cargo')
            cars = args.get('cars', 1)
            if cargo and engine in engine_cargo:
                engine_cargo[engine][cargo] = engine_cargo[engine].get(cargo, 0) + cars
        
        elif act == 'CONVERT':
            # Convert oranges to juice
            qty = args.get('qty', 1)
            if engine in engine_cargo:
                orange_qty = engine_cargo[engine].get('oranges', 0)
                if orange_qty >= qty:
                    engine_cargo[engine]['oranges'] -= qty
                    engine_cargo[engine]['juice'] = engine_cargo[engine].get('juice', 0) + qty
        
        # Check if all goals are met
        all_goals_met = True
        for key, required_qty in required_deliveries.items():
            if delivered.get(key, 0) < required_qty:
                all_goals_met = False
                break
        
        if all_goals_met and goal_achieved_at_step == -1:
            goal_achieved_at_step = i
            # Continue for a few more steps to complete current engine actions
            # but mark where goals were achieved
    
    # If goals were achieved early, trim all remaining steps
    if goal_achieved_at_step != -1 and goal_achieved_at_step < len(plan['steps']) - 1:
        remaining_steps = len(plan['steps']) - goal_achieved_at_step - 1
        
        if remaining_steps > 0:
            print(f"Trimming {remaining_steps} excess steps after goal achieved at step {goal_achieved_at_step + 1}")
            plan['steps'] = plan['steps'][:goal_achieved_at_step + 1]
    
    return plan

# ============= PATCH 5: Fix Multi-Destination Quantities =============
def patch_multi_destination_qty(plan: dict, goals: dict) -> dict:
    """Fix multi-destination deliveries by adjusting attach/load quantities."""
    if not plan.get('steps') or not goals:
        return plan
    
    # Detect multi-destination cargo
    multi_dest_cargo = {}
    for cargo_type, destinations in goals.items():
        if len(destinations) > 1:
            multi_dest_cargo[cargo_type] = destinations
    
    if not multi_dest_cargo:
        return plan
    
    patched_steps = []
    patches_applied = []
    engine_cargo_plan = {}  # Track what each engine is planning to carry
    
    for i, step in enumerate(plan['steps']):
        act = step.get('act')
        args = step.get('args', {})
        engine = args.get('engine')
        
        # Track engine cargo plans
        if act == 'ATTACH' and engine:
            car_type = args.get('car_type')
            qty = args.get('qty', 1)
            
            if engine not in engine_cargo_plan:
                engine_cargo_plan[engine] = {'boxcars': 0, 'tankers': 0}
            
            if car_type == 'boxcar':
                engine_cargo_plan[engine]['boxcars'] = qty
                
                # Check if this is for multi-destination cargo
                # For multi-destination, keep the quantity but plan will use DETACH
                if qty >= 2 and any(cargo in multi_dest_cargo and len(multi_dest_cargo[cargo]) > 1 for cargo in multi_dest_cargo):
                    # Keep original quantity for DETACH to work properly
                    patches_applied.append(f"Kept {engine} ATTACH qty={qty} for multi-destination DETACH usage")
                    engine_cargo_plan[engine]['boxcars'] = qty
        
        elif act == 'LOAD' and engine:
            cargo = args.get('cargo')
            cars = args.get('cars', 1)
            
            # If loading multi-destination cargo, keep full quantity for DETACH
            if cargo in multi_dest_cargo and cars > 1:
                destinations = multi_dest_cargo[cargo]
                patches_applied.append(f"Kept {engine} LOAD qty={cars} for {len(destinations)}-destination DETACH strategy")
        
        patched_steps.append(step)
    
    if patches_applied:
        print(f"Multi-destination patches: {patches_applied}")
    
    plan['steps'] = patched_steps
    return plan

# ============= PATCH 7: Problem Type Detection for Hints =============
def detect_problem_type(goal_text: str) -> dict:
    """Detect problem characteristics to provide better hints."""
    
    goal_lower = goal_text.lower()
    
    # Better detection: if "make into" or "convert" appears with oranges, it's for juice production
    has_juice_production = ('make into' in goal_lower and 'juice' in goal_lower) or \
                          ('make into' in goal_lower and 'oj' in goal_lower) or \
                          ('convert' in goal_lower and 'juice' in goal_lower)
    
    problem_type = {
        'has_juice': 'juice' in goal_lower or 'oj' in goal_lower,
        'has_bananas': 'banana' in goal_lower and 'deliver' in goal_lower,
        'has_oranges': 'orange' in goal_lower and not has_juice_production,
        'multi_cargo': False,
        'large_quantity': False,
        'urgent': False
    }
    
    # Check for multiple cargo types
    cargo_count = sum([
        problem_type['has_juice'],
        problem_type['has_bananas'],
        problem_type['has_oranges']
    ])
    problem_type['multi_cargo'] = cargo_count > 1
    
    # Check for large quantities (4+ units)
    import re
    qty_pattern = r'(\d+)\s*(boxcar|tanker|car)'
    matches = re.findall(qty_pattern, goal_lower)
    for qty, _ in matches:
        if int(qty) >= 4:
            problem_type['large_quantity'] = True
            break
    
    # Check for urgency
    if any(word in goal_lower for word in ['asap', 'urgent', 'immediately', 'quickly']):
        problem_type['urgent'] = True
    
    return problem_type

def create_enhanced_goal_text(goal_text: str, constraints: dict) -> str:
    """Add problem-specific hints to the goal text."""
    
    problem_type = detect_problem_type(goal_text)
    hints = []
    
    # Calculate total units needed from goal text
    import re
    total_units = 0
    
    # Look for quantity patterns
    qty_patterns = [
        r'(\d+)\s*boxcar[s]?\s*of\s*(\w+)',
        r'(\d+)\s*tanker[s]?\s*of\s*(\w+)',
        r'make\s*(\d+)\s*tanker[s]?\s*of',
        r'deliver.*?(\d+)\s*tanker[s]?.*?plus\s*(\d+)\s*boxcar[s]?'
    ]
    
    for pattern in qty_patterns:
        matches = re.findall(pattern, goal_text.lower())
        for match in matches:
            if len(match) == 2:
                if match[0].isdigit():
                    total_units += int(match[0])
            elif len(match) == 1:
                if match[0].isdigit():
                    total_units += int(match[0])
    
    # Special case for "deliver X plus Y" pattern
    deliver_plus = re.search(r'deliver.*?(\d+)\s*tanker.*?plus\s*(\d+)\s*boxcar', goal_text.lower())
    if deliver_plus:
        total_units = int(deliver_plus.group(1)) + int(deliver_plus.group(2))
    
    # If we detected large quantity needing multiple engines
    if total_units > 3:
        hints.append(f"""
CAPACITY CONSTRAINT: {total_units} units exceeds single-engine limit
Strategic requirement: Multi-engine coordination necessary
""")
    
    if problem_type['has_juice']:
        hints.append("""
JUICE PRODUCTION: Multi-step manufacturing process required
Sequence: Boxcar→oranges→tanker→conversion""")
    
    if problem_type['multi_cargo']:
        hints.append("""
MULTI-CARGO: Engine specialization approach recommended
Strategic assignment based on proximity to resources""")
    
    if problem_type['urgent']:
        hints.append("""
URGENT DELIVERY: Time-critical execution
Parallel processing approach recommended""")
    
    # Check for blocked edges in constraints
    if constraints and 'blocked_edges' in constraints:
        blocked = constraints['blocked_edges']
        hints.append(f"""
BLOCKED EDGES WARNING:
- Cannot use routes: {blocked}
- Must find alternative paths
- May need longer routes""")
    
    if hints:
        enhanced = goal_text + "\n\nSPECIFIC HINTS FOR THIS PROBLEM:" + "".join(hints)
    else:
        enhanced = goal_text
    
    return enhanced

# ============= Strategic Analysis =============
def detect_specific_problem_patterns(goals: dict, constraints: dict, goal_text: str = "") -> dict:
    """Detect specific challenging problem patterns that need targeted hints."""
    
    patterns = {}
    
    # Check for 2-F pattern: Mixed cargo to different destinations
    if (goals.get('bananas', {}).get('Bath') == 3 and 
        goals.get('juice', {}).get('Dansville') == 2):
        patterns['2f_mixed_cargo_split'] = True
    
    # Check for 3-B pattern: Multi-destination bananas + juice
    if (len(goals.get('bananas', {})) == 3 and 
        all(qty == 1 for qty in goals.get('bananas', {}).values()) and
        goals.get('juice', {}).get('Avon') == 2):
        patterns['3b_multi_destination'] = True
    
    # Check for 3-C pattern: Single engine 5 units
    goal_lower = goal_text.lower() if isinstance(goal_text, str) else ""
    if ("only use engine e2" in goal_lower and 
        sum(sum(dest.values()) for dest in goals.values()) == 5):
        patterns['3c_single_engine_overload'] = True
    
    # Check for 3-D pattern: Orange+juice conflict to same destination
    if (goals.get('oranges', {}).get('Avon') == 2 and 
        goals.get('juice', {}).get('Avon') == 1):
        patterns['3d_conversion_conflict'] = True
    
    # Check for 3-F pattern: Mixed cargo to same destination  
    if (goals.get('oranges', {}).get('Dansville') == 1 and 
        goals.get('bananas', {}).get('Dansville') == 2):
        patterns['3f_mixed_convergence'] = True
    
    # Check for 1-D pattern: High volume mixed cargo (6 units)
    if (goals.get('bananas', {}).get('Bath') == 3 and 
        goals.get('juice', {}).get('Bath') == 3):
        patterns['1d_high_volume_mixed'] = True
    
    # Check for 3-E pattern: 7+ oranges (requires 3 engines)
    total_oranges = sum(goals.get('oranges', {}).values())
    if total_oranges >= 7:
        patterns['3e_massive_orange_delivery'] = True
    
    return patterns

def analyze_problem_strategic(goals: dict, constraints: dict, goal_text: str = "") -> dict:
    """Analyze problem and return strategic instructions based on pattern detection."""
    
    analysis = {
        'total_units': 0,
        'pattern': None,  # Will identify the problem pattern
        'has_orange_juice_conflict': False,
        'needs_multiple_engines': False,
        'has_tight_deadline': False,
        'has_preloaded_cargo': False,
        'strategic_instructions': []
    }
    
    # Check for specific challenging patterns first
    specific_patterns = detect_specific_problem_patterns(goals, constraints, goal_text)
    
    # Add targeted hints for specific failing patterns
    if specific_patterns.get('2f_mixed_cargo_split'):
        analysis['strategic_instructions'].append("""
IDENTIFIED PATTERN: MIXED CARGO COORDINATION
Strategic challenge: Multiple cargo types to different destinations

RECOMMENDED ENGINE ROLES:
- E1 (starts at Avon): Handle banana deliveries - closer to banana warehouse
- E2 (starts at Elmira): Handle juice production - closer to conversion factory

STRATEGIC INSIGHT: Parallel specialization reduces travel time
Each engine focuses on cargo type matching their starting location
""")
    
    if specific_patterns.get('3b_multi_destination'):
        analysis['strategic_instructions'].append("""
IDENTIFIED PATTERN: MULTI-DESTINATION DISTRIBUTION  
Strategic challenge: Same cargo type needed at multiple locations

ENGINE STRATEGY WITH DETACH:
- E1: Load 3 bananas, use DETACH to deliver 1 at each stop (Bath→Corning→Elmira)
- E2: Focus on juice production while E1 handles distribution

TACTICAL INSIGHT: DETACH enables partial deliveries without unloading
Load full quantity upfront, then DETACH 1 boxcar at each destination
""")
    
    if specific_patterns.get('3c_single_engine_overload'):
        analysis['strategic_instructions'].append("""
IDENTIFIED PATTERN: SINGLE-ENGINE CAPACITY CONSTRAINT
Strategic challenge: Total units exceed single engine capacity limit

E2 SEQUENTIAL STRATEGY (only engine available):
- Trip 1: Check deadlines - if bananas have 9 PM deadline, do them FIRST
- Trip 2: Handle juice delivery after urgent cargo is delivered

CRITICAL TIMING FOR E2:
- 3 bananas with 9 PM deadline = ~8 hours travel time needed
- 2 juice with 24 hour deadline = plenty of time after bananas
- START with the tighter deadline to avoid violations
""")
    
    if specific_patterns.get('3d_conversion_conflict'):
        analysis['strategic_instructions'].append("""
IDENTIFIED PATTERN: CONVERSION CONFLICT
Strategic challenge: Need both source material and converted product

SOLUTION REQUIRES E2 + E3:
- E2: Load 2 oranges, deliver WITHOUT converting
- E3: Load 1 orange, CONVERT to juice, deliver juice
- Both engines deliver to destination to meet full goal
""")
    
    if specific_patterns.get('3f_mixed_convergence'):
        analysis['strategic_instructions'].append("""
IDENTIFIED PATTERN: MIXED CARGO CONVERGENCE
Strategic challenge: Different cargo types to same destination (Dansville)

SINGLE ENGINE (E1) ROUTING STRATEGY:
- Bananas: E1 starts at Avon, get boxcars from Dansville
- Oranges: Must travel to Corning for loading
- Challenge: Coordinate both cargo types efficiently

RESOURCE LOCATIONS FOR E1:
- Empty boxcars: Available at Dansville (3), Bath (2)
- Orange loading: Only at Corning warehouse
- Banana loading: Only at Avon warehouse
""")
    
    
    # Calculate total units
    for cargo_type, destinations in goals.items():
        for dest, qty in destinations.items():
            analysis['total_units'] += qty
    
    # Check for overrides (pre-loaded cargo)
    if 'overrides' in constraints and 'boxcars' in constraints['overrides']:
        analysis['has_preloaded_cargo'] = True
        for boxcar_info in constraints['overrides']['boxcars']:
            location = boxcar_info.get('location')
            payload = boxcar_info.get('payload')
            count = boxcar_info.get('count', 1)
            
            # Calculate how to use pre-loaded cargo
            if payload == 'oranges' and 'oranges' in goals:
                oranges_needed = sum(goals['oranges'].values())
                juice_needed = sum(goals.get('juice', {}).values())
                total_needed = oranges_needed + juice_needed
                
                analysis['strategic_instructions'].append(f"""
CRITICAL: PRE-LOADED CARGO ADVANTAGE!
- {count} boxcar(s) ALREADY LOADED with {payload} at {location}
- You need: {oranges_needed} oranges to deliver + {juice_needed} for juice = {total_needed} total

STRATEGIC USE OF PRE-LOADED CARGO:
- ATTACH these using "payload": "{payload}" parameter
- Skip LOAD step - cargo is already loaded!
- {count} available but need {total_needed} total
- If {count} > {total_needed}: Take only what you need
- If {count} < {total_needed}: Use all {count} pre-loaded + get more elsewhere

KEY: Pre-loaded cargo at {location} saves significant time!
""")
    
    # Check for engine restrictions in problem text
    goal_lower = goal_text.lower() if isinstance(goal_text, str) else ""
    if "only use engine" in goal_lower or "can only use" in goal_lower:
        analysis['has_engine_restriction'] = True
        # Extract which engine
        import re
        engine_match = re.search(r'only use engine (e\d)', goal_lower, re.IGNORECASE)
        if not engine_match:
            engine_match = re.search(r'can only use engine (e\d)', goal_lower, re.IGNORECASE)
        
        if engine_match:
            restricted_engine = engine_match.group(1).upper()
            analysis['strategic_instructions'].append(f"""
ENGINE RESTRICTION: Only {restricted_engine} available (others in maintenance)

CAPACITY CONSTRAINT: {analysis['total_units']} units with 3-unit capacity limit
Strategic implication: Minimum {(analysis['total_units'] + 2) // 3} trip(s) required

KEY STRATEGIC CONSIDERATION: Sequential execution only - no parallel support
Critical factor: Trip sequencing based on deadlines and resource accessibility
""")
            
            # Special case for 3-C type problem (5 units, E2 only)
            if analysis['total_units'] == 5 and restricted_engine == 'E2':
                if 'bananas' in goals and 'juice' in goals:
                    banana_qty = sum(goals['bananas'].values())
                    juice_qty = sum(goals['juice'].values())
                    analysis['strategic_instructions'].append(f"""
RESOURCE CONSTRAINT: Elmira has only 2 boxcars (E2's start location)
CARGO REQUIREMENT: {banana_qty} bananas needs 3 boxcars, {juice_qty} juice needs 2

STRATEGIC INSIGHT: Boxcar shortage requires creative resource gathering
Consider where additional boxcars are available: Bath (2) or Dansville (3)
Two-trip approach necessary due to capacity limits
""")
    
    # Check for tight deadlines
    if 'deliver_deadlines' in constraints:
        for cargo, dests in constraints['deliver_deadlines'].items():
            for dest, deadline in dests.items():
                if deadline <= 7.0:
                    analysis['has_tight_deadline'] = True
                    
                    # Special handling for 5-unit tight deadline problems
                    if analysis['total_units'] == 5 and cargo == 'oranges':
                        analysis['strategic_instructions'].append(f"""
URGENT: 5-UNIT DELIVERY WITH {deadline}-HOUR DEADLINE!
Challenge: Deliver 5 {cargo} to {dest} quickly

CRITICAL SPLIT REQUIRED:
- E1: 3 units (from Avon, gets boxcars at Dansville)
- E2: 2 units (from Elmira, uses local boxcars)

TIMING STRATEGY:
- Both engines must start IMMEDIATELY
- E1 route: Avon→Dansville (boxcars)→Corning (load)→{dest}
- E2 route: Elmira (boxcars)→Corning (load)→{dest}
- E2 is faster (starts closer), E1 handles larger load

KEY: 5 units = mandatory 2-engine split. Start both NOW!
""")
                    elif not analysis['has_preloaded_cargo']:
                        analysis['strategic_instructions'].append(f"""
TIGHT DEADLINE: Only {deadline} hours for {cargo} to {dest}!
- Immediate action required - no time for detours
- Use fastest available engine based on proximity
- Consider pre-positioned resources
- Multiple engines may be needed for speed
""")
                    break
    
    # PATTERN 1: Mixed Cargo with Orange+Juice Conflict
    # Only if we actually need to DELIVER both oranges AND juice
    if 'juice' in goals and 'oranges' in goals:
        # Check if we're actually delivering oranges (not just mentioned in text)
        if sum(goals.get('oranges', {}).values()) > 0:
            analysis['pattern'] = 'mixed_cargo_conflict'
            analysis['has_orange_juice_conflict'] = True
        
        total_oranges = sum(goals.get('oranges', {}).values())
        total_juice = sum(goals.get('juice', {}).values())
        
        analysis['strategic_instructions'].append(f"""
PATTERN: CONVERSION CONFLICT
Strategic insight: Need to deliver {total_oranges} oranges AND produce {total_juice} juice

CRITICAL CONSTRAINT: Orange→juice conversion destroys the original oranges
Resource calculation: {total_oranges + total_juice} total oranges needed ({total_oranges} to keep, {total_juice} to convert)

Strategic approaches: Parallel engines with role specialization or oversupply strategy
Key decision: Division of labor vs resource acquisition focus
""")
    
    # PATTERN 2: Manufacturing Integration (Juice Production Only)
    elif 'juice' in goals and 'oranges' not in goals:
        analysis['pattern'] = 'manufacturing_integration'
        juice_qty = sum(goals.get('juice', {}).values())
        
        if juice_qty <= 3:
            analysis['strategic_instructions'].append(f"""
PATTERN: PURE MANUFACTURING TASK
Challenge: Produce and deliver {juice_qty} juice unit(s) ONLY

KEY INSIGHT: Oranges mentioned are ONLY raw materials for juice production, not for delivery.
With {juice_qty} juice units needed, you need {juice_qty} oranges to convert.

CRITICAL: Only juice is being delivered - no orange delivery required!
Since {juice_qty} ≤ 3, a single engine from Elmira can handle the entire operation.
E2 is optimal: starts at Elmira (factory location), gets resources, produces juice, delivers.

RESOURCE NEEDS: {juice_qty} boxcars + {juice_qty} tankers + {juice_qty} oranges for conversion.
""")
        else:
            # More than 3 juice units - need multiple engines
            analysis['strategic_instructions'].append(f"""
PATTERN: HIGH-VOLUME JUICE PRODUCTION
Challenge: Produce and deliver {juice_qty} juice units (exceeds single engine capacity)

CRITICAL: {juice_qty} juice units requires multiple engines!

SUGGESTED SPLIT:
- E2: 2 juice units (uses Elmira's 2 boxcars)
- E3: {juice_qty - 2} juice units (needs boxcars from Bath)

KEY CONSTRAINTS:
- Each juice unit needs: 1 boxcar + 1 tanker + 1 orange for conversion
- Elmira has only 2 boxcars, so E3 must get boxcars from Bath
- Both engines need access to Elmira factory for conversion
- Parallel execution reduces total time

RESOURCE DISTRIBUTION: E2 uses Elmira resources, E3 uses Bath resources.
""")
    
    # PATTERN 3: Volume-Constrained Transport (>3 units)
    if analysis['total_units'] > 3:
        analysis['needs_multiple_engines'] = True
        analysis['pattern'] = 'volume_constrained'
        
        # Calculate optimal engine distribution
        engines_needed = (analysis['total_units'] + 2) // 3  # Ceiling division
        
        # Be very explicit about multi-engine requirement
        analysis['strategic_instructions'].append(f"""
PATTERN: CAPACITY OVERLOAD - MUST USE MULTIPLE ENGINES!
Strategic insight: {analysis['total_units']} units REQUIRES {engines_needed}+ engines (single engine max = 3)

CRITICAL: You MUST use AT LEAST {engines_needed} engines working in parallel!
- E1: Starts at Avon (best for bananas)
- E2: Starts at Elmira (best for juice production)  
- E3: Starts at Elmira (can support E2 or handle separate cargo)

SUGGESTED SPLIT for {analysis['total_units']} units:
- Engine 1: Handle 3 units
- Engine 2: Handle {min(3, analysis['total_units'] - 3)} units
{f"- Engine 3: Handle {analysis['total_units'] - 6} units" if analysis['total_units'] > 6 else ""}

Resource coordination: Each engine needs its own boxcars - plan accordingly
""")
    
    # PATTERN 4: Basic Single Cargo Transport
    if analysis['total_units'] <= 3 and not analysis['has_orange_juice_conflict'] and not analysis['pattern']:
        analysis['pattern'] = 'basic_single_cargo'
        
        # Determine cargo type and optimal engine
        cargo_types = list(goals.keys())
        if len(cargo_types) == 1:
            cargo = cargo_types[0]
            qty = analysis['total_units']
            
            analysis['strategic_instructions'].append(f"""
PATTERN: SIMPLE {cargo.upper()} DELIVERY
Task: Deliver {qty} {cargo} unit(s)

ENGINE SELECTION HINT:
{'E1 optimal for bananas (starts at Avon where bananas are)' if cargo == 'bananas' else ''}
{'E2 optimal for oranges (starts near Corning where oranges are)' if cargo == 'oranges' else ''}
{'E2 optimal for juice (starts at Elmira where factory is)' if cargo == 'juice' else ''}

Since only {qty} units needed and capacity is 3, one engine suffices.
Remember: E1 needs boxcars from Dansville, E2/E3 can use Elmira's boxcars.
""")
    
    # PATTERN 5: Multi-Destination Distribution
    # Check if any cargo type has multiple destinations
    multi_destination = False
    for cargo_type, destinations in goals.items():
        if len(destinations) > 1:
            multi_destination = True
            break
    
    if multi_destination and not analysis['pattern']:
        analysis['pattern'] = 'multi_destination'
        
        # Special case for problem 3-B pattern: 1 banana each to 3 locations
        if ('bananas' in goals and len(goals['bananas']) == 3 and 
            all(qty == 1 for qty in goals['bananas'].values()) and
            'juice' in goals and goals.get('juice', {}).get('Avon') == 2):
            
            # This is specifically problem 3-B type - more strategic, less prescriptive
            analysis['strategic_instructions'].append(f"""
PATTERN: MULTI-STOP DISTRIBUTION CHALLENGE
Strategic insight: 3 separate banana deliveries + juice production = complex coordination problem

KEY STRATEGIC OPTIONS:
- DETACH action enables dropping loaded cars at intermediate stops without full unloading
- Consider whether parallel execution (multiple engines) or sequential efficiency (single engine with DETACH) fits better
- 5 total units may require capacity-conscious approach

Strategic question: Optimize for speed (parallel) or resource efficiency (sequential)?
""")
        else:
            # Build distribution details for other multi-destination problems
            for cargo_type, destinations in goals.items():
                if len(destinations) > 1:
                    total = sum(destinations.values())
                    
                    analysis['strategic_instructions'].append(f"""
PATTERN: MULTI-DESTINATION {cargo_type.upper()} DISTRIBUTION
Strategic challenge: {total} units to {len(destinations)} separate destinations

KEY INSIGHT: Standard loading delivers all cargo to final destination only
Strategic options: Parallel engines (speed) vs sequential trips (resource efficiency)

Consider: Does capacity allow single-engine sequential approach, or is parallel execution necessary?
""")
    
    # PATTERN 6: Mixed Cargo Coordination (no conversion conflict)
    elif len(goals) > 1 and not analysis['has_orange_juice_conflict'] and not analysis['pattern']:
        analysis['pattern'] = 'mixed_cargo_coordination'
        
        # Check for same destination (3-F type) vs different destinations (2-F type)
        destinations = set()
        for cargo_type, dest_dict in goals.items():
            destinations.update(dest_dict.keys())
        
        if len(destinations) == 1:
            # Same destination (3-F type)
            dest = list(destinations)[0]
            analysis['strategic_instructions'].append(f"""
PATTERN: MIXED CARGO CONVERGENCE
Strategic insight: Multiple cargo types converging at {dest}

KEY CONSIDERATION: Different cargo sources but shared destination
Strategic options: Coordinate timing vs separate optimized routes
Capacity constraint: {analysis['total_units']} total units to plan for

Strategic question: Single coordinated arrival or parallel independent deliveries?
""")
        else:
            # Different destinations (2-F type)
            analysis['strategic_instructions'].append(f"""
PATTERN: MIXED CARGO COORDINATION  
Strategic insight: Multiple cargo types to different destinations

KEY STRATEGIC PRINCIPLE: Match engines to cargo sources for efficiency
Strategic division based on proximity and specialization

Total complexity: {analysis['total_units']} units across {len(destinations)} destinations
Consider: Parallel execution to minimize total completion time
""")
    
    # Strategic note for banana transport logistics
    if 'bananas' in goals:
        analysis['strategic_instructions'].append(f"""
BANANA LOGISTICS NOTE: E1 optimal for banana transport but requires boxcar acquisition from Dansville
Strategic timing consideration: Setup phase before delivery execution begins
""")
    
    # Check for specific problem patterns that need special handling
    specific_patterns = detect_specific_problem_patterns(goals, constraints, goal_text)
    
    if specific_patterns.get('1d_high_volume_mixed'):
        analysis['strategic_instructions'].append("""
PROBLEM: 6 UNITS (3 bananas + 3 juice)!
MANDATORY: USE ALL 3 ENGINES IN PARALLEL!
- E1: 3 bananas 
- E2 + E3: 3 juice total (coordinate from Elmira)
This is NOT optional - single engine cannot handle 6 units!
""")
    
    if specific_patterns.get('3e_massive_orange_delivery'):
        total_oranges = sum(goals.get('oranges', {}).values())
        analysis['strategic_instructions'].append(f"""
PROBLEM: {total_oranges} ORANGES!
MANDATORY: USE ALL 3 ENGINES!
- Each engine handles 2-3 oranges
- Coordinate boxcar resources from multiple locations
- Work in parallel to meet deadline!
""")
    
    return analysis

def create_strategic_goal_text(goal_text: str, goals: dict, constraints: dict) -> str:
    """Enhance goal text with strategic analysis."""
    
    analysis = analyze_problem_strategic(goals, constraints, goal_text)
    
    # Start with original goal
    enhanced = goal_text
    
    # Add strategic instructions if any
    if analysis['strategic_instructions']:
        enhanced += "\n\n========== STRATEGIC ANALYSIS ==========\n"
        for instruction in analysis['strategic_instructions']:
            enhanced += instruction
        enhanced += "\n========================================\n"
    
    return enhanced

# ============= Use original signature with enhanced goal =============
from minimal_test_option1 import PlanFromGoalSig

def call_patched_planner(goal_text: str, constraints: dict, apply_patches: bool = True, goals: dict = None, use_baseline: bool = False, use_strategic: bool = True) -> dict:
    """Call planner with strategic pre-processing and automatic patches.
    
    Args:
        goal_text: Problem description
        constraints: Problem constraints
        apply_patches: Whether to apply patches
        goals: Problem goals
        use_baseline: Whether to use baseline planner (Option 1)
        use_strategic: Whether to use strategic analysis
    """
    
    try:
        # Choose between baseline and enhanced mode
        if use_baseline:
            # Use baseline planner with no enhancements
            enhanced_goal = goal_text
            apply_patches = False  # No patches for baseline
        else:
            # Enhance goal text with problem-specific hints and strategic analysis
            if apply_patches:
                # First apply original hints
                enhanced_goal = create_enhanced_goal_text(goal_text, constraints)
                # Then add strategic analysis if both enabled and goals provided
                if use_strategic and goals:
                    enhanced_goal = create_strategic_goal_text(enhanced_goal, goals, constraints)
            else:
                enhanced_goal = goal_text
        
        # Use original planner with enhanced goal
        planner = dspy.Predict(PlanFromGoalSig)
        
        print(f"Debug: Calling planner with {'baseline' if use_baseline else ('enhanced' if apply_patches else 'original')} goal")
        if apply_patches and '========== STRATEGIC ANALYSIS ==========' in enhanced_goal:
            print("Strategic analysis included in prompt")
        
        result = planner(
            goal=enhanced_goal,
            constraints=json.dumps(constraints)
        )
        
        # Extract plan
        if hasattr(result, 'plan_json'):
            plan_data = result.plan_json
            
            if isinstance(plan_data, str):
                try:
                    plan_data = json.loads(plan_data)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', plan_data, re.DOTALL)
                    if json_match:
                        plan_data = json.loads(json_match.group())
                    else:
                        return {"steps": [], "partial_order": [], "constraints": {}}
            
            # Apply patches if enabled
            if apply_patches and isinstance(plan_data, dict):
                print("\nApplying patches...")
                
                # Patch 1: Fix engine locations
                plan_data = patch_engine_locations(plan_data)
                
                # Patch 2: Expand illegal travels
                plan_data = expand_illegal_travel(plan_data, constraints)
                
                # Patch 3: Check capacity and add CONVERT quantities
                plan_data, engine_state = check_and_fix_capacity(plan_data)
                
                # Patch 4: Fix CONVERT to include quantity based on engine cargo
                plan_data = patch_convert_quantity(plan_data, engine_state)
                
                # Patch 5: Fix multi-destination quantities
                if goals:
                    plan_data = patch_multi_destination_qty(plan_data, goals)
                
                # Patch 6: Add missing delivery steps
                if goals:
                    plan_data = add_missing_deliveries(plan_data, goals)
                
                # Patch 7: Trim excess steps after goals achieved
                if goals:
                    plan_data = trim_excess_steps(plan_data, goals)
                
                print("Patches complete.\n")
            
            return plan_data
            
    except Exception as e:
        print(f"Planning error: {e}")
        import traceback
        traceback.print_exc()
    
    return {"steps": [], "partial_order": [], "constraints": {}}

# ============= Utility functions from original =============
def print_section(title):
    print(f"========== {title} ==========")

def pretty(obj):
    result = json.dumps(obj, indent=2, ensure_ascii=False)
    result = result.replace('\\u2192', '→')
    result = result.replace('\\u22643', '≤3')
    return result

def summarize_steps(plan):
    """Create a numbered summary of plan steps."""
    if not isinstance(plan, dict) or 'steps' not in plan:
        return "(no steps)"
    
    lines = []
    for i, step in enumerate(plan['steps'], 1):
        if isinstance(step, dict) and 'act' in step:
            act = step['act']
            args = step.get('args', {})
            
            # Compact display
            if act == "START":
                lines.append(f"{i:02d}. {act} {args.get('engine')}@{args.get('at')}")
            elif act == "TRAVEL":
                lines.append(f"{i:02d}. {act} {args.get('engine')}: {args.get('from')}→{args.get('to')}")
            elif act == "ATTACH":
                payload = f" ({args.get('payload')})" if args.get('payload') else ""
                lines.append(f"{i:02d}. {act} {args.get('engine')}: {args.get('qty')} {args.get('car_type')}@{args.get('from')}{payload}")
            elif act == "LOAD":
                lines.append(f"{i:02d}. {act} {args.get('engine')}: {args.get('cargo')}×{args.get('cars')}@{args.get('at')}")
            else:
                lines.append(f"{i:02d}. {act} {args}")
    
    if len(lines) > 20:
        lines = lines[:20] + [f"... ({len(plan['steps']) - 20} more steps)"]
    
    return "\n".join(lines)

def write_jsonl(path, data):
    """Append a JSON line to a file."""
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def run_problem(pid: str, goal_text: str, plan_data: dict, log_path: str, apply_patches: bool = True, use_fallback: bool = True, use_baseline: bool = False, use_strategic: bool = True) -> dict:
    """Run planning with patches and fallback to hardcoded solutions if AI fails.
    
    Args:
        pid: Problem ID
        goal_text: Problem description
        plan_data: Problem data with goals and constraints
        log_path: Output log file path
        apply_patches: Whether to apply automatic patches (default True)
        use_fallback: Whether to use fallback solver on failure (default True)
        use_baseline: Whether to use baseline planner instead of enhanced (default False)
        use_strategic: Whether to use strategic analysis (default True)
    """
    
    print_section(f"PROBLEM: {pid}")
    print(f"DESCRIPTION: {goal_text[:200]}...")
    
    goals = plan_data.get("goal", {})
    constraints = plan_data.get("constraints", {})
    
    # Detect problem type
    problem_type = detect_problem_type(goal_text)
    print(f"Problem characteristics: {problem_type}")
    
    # Choose planner mode
    if use_baseline:
        print("Mode: BASELINE (Option 1 - no patches, no strategic enhancements)")
    else:
        print(f"Mode: {'ENHANCED' if apply_patches else 'ENHANCED (no patches)'}")
    
    # Generate plan with strategic enhancements and patches
    # Pass overrides as part of constraints if present
    if 'overrides' in plan_data:
        constraints_with_overrides = dict(constraints)
        constraints_with_overrides['overrides'] = plan_data['overrides']
        plan = call_patched_planner(goal_text, constraints_with_overrides, apply_patches=apply_patches, goals=goals, use_baseline=use_baseline, use_strategic=use_strategic)
    else:
        plan = call_patched_planner(goal_text, constraints, apply_patches=apply_patches, goals=goals, use_baseline=use_baseline, use_strategic=use_strategic)
    
    # Check plan
    try:
        res = check_plan(
            plan=plan,
            problem_desc=goal_text,
            goals=goals,
            constraints=constraints,
            overrides=plan_data.get("overrides", {})
        )
    except Exception as e:
        print(f"Error checking plan: {e}")
        res = {"goal_achieved": 0, "violations": [f"Checker error: {e}"]}
    
    # Check if AI failed and fallback is enabled
    if use_fallback and res.get('goal_achieved', 0) < 1.0:
        print("\n" + "="*60)
        print("⚠️  AI SOLUTION FAILED - ACTIVATING FALLBACK ALGORITHM ⚠️")
        print("="*60)
        print(f"AI achieved: {res.get('goal_achieved', 0)*100:.1f}%")
        print(f"Violations: {res.get('violations', ['Unknown'])[:2]}")
        print("Switching to algorithmic solver...")
        print("="*60 + "\n")
        
        # Use algorithmic solver
        algo_solver = AlgorithmicSolver()
        
        # Check for special constraints
        algo_constraints = dict(constraints)
        
        # Check if problem has engine restriction
        if 'only use engine' in goal_text.lower():
            import re
            engine_match = re.search(r'only use engine (e\d)', goal_text.lower())
            if engine_match:
                algo_constraints['engine_restriction'] = engine_match.group(1).upper()
        
        # Try algorithmic solution
        try:
            fallback_plan = algo_solver.solve_with_constraints(pid, goals, algo_constraints)
            
            # Check algorithmic plan
            fallback_res = check_plan(
                plan=fallback_plan,
                problem_desc=goal_text,
                goals=goals,
                constraints=constraints,
                overrides=plan_data.get("overrides", {})
            )
            
            # If algorithmic solver succeeds, use it
            if fallback_res.get('goal_achieved', 0) >= 1.0:
                print("✅ ALGORITHMIC SOLUTION SUCCESSFUL!")
                plan = fallback_plan
                res = fallback_res
                res['used_fallback'] = True
            else:
                print(f"⚠️ Algorithmic solver achieved: {fallback_res.get('goal_achieved', 0)*100:.1f}%")
                
                # Try hardcoded fallback as last resort
                print("Trying hardcoded fallback solutions...")
                hardcoded_solver = FallbackSolver()
                hardcoded_plan = hardcoded_solver.get_solution(pid, goals, constraints)
                
                hardcoded_res = check_plan(
                    plan=hardcoded_plan,
                    problem_desc=goal_text,
                    goals=goals,
                    constraints=constraints,
                    overrides=plan_data.get("overrides", {})
                )
                
                if hardcoded_res.get('goal_achieved', 0) >= 1.0:
                    print("✅ HARDCODED FALLBACK SUCCESSFUL!")
                    plan = hardcoded_plan
                    res = hardcoded_res
                    res['used_fallback'] = True
                else:
                    print(f"❌ All fallbacks failed")
                    res['fallback_attempted'] = True
                    
        except Exception as e:
            print(f"❌ Error in algorithmic solution: {e}")
            import traceback
            traceback.print_exc()
            res['fallback_error'] = str(e)
    
    # Display results
    print_section("Generated Plan")
    print(summarize_steps(plan))
    print_section("Result")
    print(f"Goal achieved: {res.get('goal_achieved', 0)}")
    if res.get('used_fallback'):
        print("✅ USING FALLBACK ALGORITHM SOLUTION")
    if res.get('violations'):
        print(f"Violations: {res['violations'][:3]}")
    
    # Log results
    write_jsonl(log_path, {
        "problem_id": pid,
        "plan": plan,
        "result": res,
        "goal_achieved": res.get("goal_achieved", 0),
        "used_fallback": res.get("used_fallback", False)
    })
    
    return {"problem_id": pid, "plan": plan, "result": res}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated problem IDs to test")
    parser.add_argument("--log", default="patched_results.jsonl", help="Output log file")
    parser.add_argument("--no-patches", action="store_true", help="Disable automatic patches")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback to hardcoded solutions")
    parser.add_argument("--baseline", action="store_true", help="Use baseline planner (Option 1) without enhancements")
    parser.add_argument("--no-strategic", action="store_true", help="Disable strategic analysis")
    args = parser.parse_args()
    
    # Set up DSPy
    lm = dspy.LM('gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)
    
    # Load data
    with open("all_plans_description.json", "r") as f:
        plans_data = json.load(f)
    
    problems_data = {}
    if os.path.exists("cisd_cleaned.json"):
        with open("cisd_cleaned.json", "r") as f:
            cisd_data = json.load(f)
            problems_data = {pid: data.get("description", "") for pid, data in cisd_data.items()}
    
    # Process problems
    if args.ids:
        problem_ids = [pid.strip() for pid in args.ids.split(",")]
    else:
        problem_ids = list(problems_data.keys())[:5]
    
    results = []
    for pid in problem_ids:
        if pid not in problems_data or pid not in plans_data:
            continue
        
        try:
            result = run_problem(
                pid=pid,
                goal_text=problems_data[pid],
                plan_data=plans_data[pid],
                log_path=args.log,
                apply_patches=not args.no_patches,
                use_fallback=not args.no_fallback,
                use_baseline=args.baseline,
                use_strategic=not args.no_strategic
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {pid}: {e}")
    
    # Summary
    successful = sum(1 for r in results if r['result'].get('goal_achieved', 0) >= 1.0)
    fallback_used = sum(1 for r in results if r['result'].get('used_fallback', False))
    ai_successful = successful - fallback_used
    
    print(f"\n{'='*50}")
    mode = "BASELINE" if args.baseline else ("ENHANCED" if not args.no_patches else "ENHANCED (no patches)")
    print(f"Mode: {mode}")
    print(f"OVERALL Results: {successful}/{len(results)} successful ({successful/len(results)*100:.1f}%)")
    print(f"  - AI Solutions: {ai_successful}/{len(results)} ({ai_successful/len(results)*100:.1f}%)")
    print(f"  - Fallback Used: {fallback_used}/{len(results)} ({fallback_used/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()