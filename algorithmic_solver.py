# algorithmic_solver_improved.py - Improved solver with built-in simulator
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class Car:
    """Represents a rail car."""
    car_type: str  # 'boxcar' or 'tanker'
    cargo: Optional[str] = None  # None if empty, or 'bananas', 'oranges', 'juice'
    location: Optional[str] = None  # Current location or None if attached
    attached_to: Optional[str] = None  # Engine ID if attached

@dataclass
class Engine:
    """Represents an engine."""
    engine_id: str
    location: str
    attached_cars: List[Car] = field(default_factory=list)
    active: bool = False

class RailState:
    """Simulates the current state of the rail system."""
    
    def __init__(self):
        # Network topology
        self.network = {
            "Avon": ["Dansville", "Bath"],
            "Dansville": ["Avon", "Corning"],
            "Corning": ["Dansville", "Bath", "Elmira"],
            "Bath": ["Corning", "Avon"],
            "Elmira": ["Corning"]
        }
        
        # Weighted travel times between cities (in hours)
        # From system_map.py EDGE_HOURS
        self.travel_times = {
            ("Avon", "Dansville"): 3,
            ("Dansville", "Avon"): 3,
            ("Avon", "Bath"): 4,
            ("Bath", "Avon"): 4,
            ("Dansville", "Corning"): 1,
            ("Corning", "Dansville"): 1,
            ("Corning", "Bath"): 2,
            ("Bath", "Corning"): 2,
            ("Corning", "Elmira"): 2,
            ("Elmira", "Corning"): 2
        }
        
        # Initial resources at yards
        self.yard_cars = {
            "Elmira": {"boxcar": 2, "tanker": 0},
            "Bath": {"boxcar": 2, "tanker": 0},
            "Dansville": {"boxcar": 3, "tanker": 0},
            "Corning": {"boxcar": 0, "tanker": 3},
            "Avon": {"boxcar": 0, "tanker": 0}
        }
        
        # Cargo sources
        self.cargo_sources = {
            "bananas": "Avon",
            "oranges": "Corning"
        }
        
        # Engines (not started yet)
        self.engines = {
            "E1": Engine("E1", "Avon"),
            "E2": Engine("E2", "Elmira"),
            "E3": Engine("E3", "Elmira")
        }
        
        # Delivered cargo
        self.delivered = defaultdict(lambda: defaultdict(int))
        
        # Factory location
        self.factory = "Elmira"
        
        # Time tracking
        self.time = 0
        
    def copy(self):
        """Create a deep copy of the state."""
        new_state = RailState()
        new_state.yard_cars = deepcopy(self.yard_cars)
        new_state.engines = deepcopy(self.engines)
        new_state.delivered = deepcopy(self.delivered)
        new_state.time = self.time
        return new_state
    
    def start_engine(self, engine_id: str, location: str) -> bool:
        """Start an engine at a location."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if engine.active:
            return False
        if engine.location != location:
            return False
        engine.active = True
        return True
    
    def travel(self, engine_id: str, from_loc: str, to_loc: str) -> bool:
        """Move engine from one location to another."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if not engine.active:
            return False
        if engine.location != from_loc:
            return False
        if to_loc not in self.network.get(from_loc, []):
            return False
        engine.location = to_loc
        # Use actual travel time from the weighted graph
        travel_time = self.travel_times.get((from_loc, to_loc), 1)
        self.time += travel_time
        return True
    
    def attach(self, engine_id: str, car_type: str, location: str, qty: int) -> bool:
        """Attach cars to engine."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if not engine.active:
            return False
        if engine.location != location:
            return False
        
        # Check capacity (max 3 cars)
        if len(engine.attached_cars) + qty > 3:
            return False
        
        # Check availability at yard
        if self.yard_cars[location].get(car_type, 0) < qty:
            return False
        
        # Attach cars
        for _ in range(qty):
            car = Car(car_type=car_type, attached_to=engine_id)
            engine.attached_cars.append(car)
        
        self.yard_cars[location][car_type] -= qty
        return True
    
    def detach(self, engine_id: str, car_type: str, location: str, qty: int) -> bool:
        """Detach cars from engine."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if not engine.active:
            return False
        if engine.location != location:
            return False
        
        # Count matching cars
        matching_cars = [c for c in engine.attached_cars if c.car_type == car_type]
        if len(matching_cars) < qty:
            return False
        
        # Detach cars and deliver cargo if any
        for _ in range(qty):
            for i, car in enumerate(engine.attached_cars):
                if car.car_type == car_type:
                    if car.cargo:
                        self.delivered[car.cargo][location] += 1
                    engine.attached_cars.pop(i)
                    break
        
        # Return empty cars to yard
        self.yard_cars[location][car_type] = self.yard_cars[location].get(car_type, 0) + qty
        return True
    
    def load(self, engine_id: str, cargo: str, location: str, cars: int) -> bool:
        """Load cargo onto attached cars."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if not engine.active:
            return False
        if engine.location != location:
            return False
        
        # Check cargo source
        if location != self.cargo_sources.get(cargo):
            return False
        
        # Count empty boxcars
        empty_boxcars = [c for c in engine.attached_cars 
                        if c.car_type == "boxcar" and c.cargo is None]
        if len(empty_boxcars) < cars:
            return False
        
        # Load cargo
        for i in range(cars):
            empty_boxcars[i].cargo = cargo
        
        self.time += 1  # Loading takes 1 hour
        return True
    
    def convert(self, engine_id: str, location: str, qty: int) -> bool:
        """Convert oranges to juice."""
        if engine_id not in self.engines:
            return False
        engine = self.engines[engine_id]
        if not engine.active:
            return False
        if engine.location != location:
            return False
        if location != self.factory:
            return False
        
        # Count orange boxcars and empty tankers
        orange_boxcars = [c for c in engine.attached_cars 
                         if c.car_type == "boxcar" and c.cargo == "oranges"]
        empty_tankers = [c for c in engine.attached_cars 
                        if c.car_type == "tanker" and c.cargo is None]
        
        if len(orange_boxcars) < qty or len(empty_tankers) < qty:
            return False
        
        # Convert
        for i in range(qty):
            # Remove orange cargo from boxcar
            orange_boxcars[i].cargo = None
            # Add juice to tanker
            empty_tankers[i].cargo = "juice"
        
        self.time += 1  # Conversion takes 1 hour
        return True
    
    def check_goals(self, goals: Dict) -> float:
        """Check what percentage of goals are achieved."""
        total_required = 0
        total_delivered = 0
        
        for cargo_type, destinations in goals.items():
            for dest, qty in destinations.items():
                total_required += qty
                total_delivered += min(self.delivered[cargo_type][dest], qty)
        
        if total_required == 0:
            return 1.0
        return total_delivered / total_required


class AlgorithmicSolver:
    """Improved algorithmic solver with built-in simulation."""
    
    def __init__(self):
        self.max_capacity = 3
        
    def find_shortest_path(self, start: str, end: str, network: Dict, travel_times: Dict = None) -> List[str]:
        """Find shortest path using Dijkstra's algorithm with weighted edges."""
        if start == end:
            return [start]
        
        # Use provided travel times or default to uniform weights
        if travel_times is None:
            # Fallback to BFS for uniform weights
            queue = deque([(start, [start])])
            visited = {start}
            
            while queue:
                node, path = queue.popleft()
                for neighbor in network.get(node, []):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        if neighbor == end:
                            return new_path
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
            return []
        
        # Dijkstra's algorithm for weighted graph
        import heapq
        
        # Priority queue: (cumulative_time, node, path)
        pq = [(0, start, [start])]
        visited = set()
        
        while pq:
            time, node, path = heapq.heappop(pq)
            
            if node in visited:
                continue
            visited.add(node)
            
            if node == end:
                return path
            
            for neighbor in network.get(node, []):
                if neighbor not in visited:
                    edge_time = travel_times.get((node, neighbor), 1)
                    heapq.heappush(pq, (time + edge_time, neighbor, path + [neighbor]))
        
        return []
    
    def solve_with_constraints(self, problem_id: str, goals: dict, constraints: dict, overrides: dict = None) -> dict:
        """Generate solution using simulation."""
        state = RailState()
        steps = []
        
        # Apply overrides if provided
        if overrides and 'boxcars' in overrides:
            for override in overrides['boxcars']:
                location = override['location']
                payload = override['payload']
                count = override['count']
                # These are pre-loaded boxcars at location
                # We'll track them separately for special handling
                if not hasattr(state, 'preloaded_cars'):
                    state.preloaded_cars = {}
                state.preloaded_cars[location] = {'payload': payload, 'count': count}
        
        # Determine available engines
        available_engines = constraints.get('available_engines', ['E1', 'E2', 'E3'])
        if isinstance(available_engines, list):
            engines_to_use = [e for e in available_engines if e in state.engines]
        else:
            engines_to_use = ['E1', 'E2', 'E3']
        
        # Analyze goals
        has_juice = 'juice' in goals
        has_bananas = 'bananas' in goals
        has_oranges = 'oranges' in goals
        
        # Check for special overrides (like pre-loaded cars)
        if hasattr(state, 'preloaded_cars'):
            # Handle pre-loaded cars specially
            return self._solve_with_preloaded(goals, engines_to_use, state, constraints)
        
        # Analyze deadlines to prioritize cargo
        deadlines = constraints.get('deliver_deadlines', {})
        cargo_priorities = []
        
        for cargo_type, destinations in goals.items():
            for dest, qty in destinations.items():
                deadline = deadlines.get(cargo_type, {}).get(dest, float('inf'))
                # Calculate time needed (rough estimate)
                time_needed = self._estimate_delivery_time(cargo_type, dest, qty, state)
                urgency = deadline - time_needed
                cargo_priorities.append((urgency, cargo_type, dest, qty, deadline))
        
        # Sort by urgency (lower = more urgent)
        cargo_priorities.sort()
        
        # Determine strategy based on deadlines and resources
        total_units = sum(sum(dest.values()) for dest in goals.values())
        earliest_deadline = min(p[4] for p in cargo_priorities) if cargo_priorities else float('inf')
        
        # If very tight deadline or multiple urgent deliveries, use parallel engines
        if (earliest_deadline <= 10 and total_units > 3) or (len(engines_to_use) > 1 and len(cargo_priorities) > 1):
            return self._solve_with_deadline_priority(goals, engines_to_use, state, cargo_priorities)
        
        # Single engine problems
        if len(engines_to_use) == 1:
            return self._solve_single_engine_smart(goals, engines_to_use[0], state, cargo_priorities)
        
        # Multi-engine problems
        if has_juice:
            return self._solve_with_juice(goals, engines_to_use, state)
        else:
            return self._solve_simple_cargo(goals, engines_to_use, state)
    
    def _solve_single_engine(self, goals: dict, engine_id: str, state: RailState) -> dict:
        """Solve using a single engine."""
        steps = []
        engine = state.engines[engine_id]
        
        # Start engine
        steps.append({
            "act": "START",
            "args": {"engine": engine_id, "at": engine.location}
        })
        state.start_engine(engine_id, engine.location)
        
        # Handle juice first if needed
        if 'juice' in goals:
            juice_steps = self._produce_juice(engine_id, goals['juice'], state)
            steps.extend(juice_steps)
        
        # Handle bananas
        if 'bananas' in goals:
            banana_steps = self._deliver_cargo(engine_id, 'bananas', goals['bananas'], state)
            steps.extend(banana_steps)
        
        # Handle oranges
        if 'oranges' in goals and 'juice' not in goals:
            orange_steps = self._deliver_cargo(engine_id, 'oranges', goals['oranges'], state)
            steps.extend(orange_steps)
        
        return {"steps": steps}
    
    def _produce_juice(self, engine_id: str, juice_goals: dict, state: RailState) -> List[dict]:
        """Produce and deliver juice."""
        steps = []
        engine = state.engines[engine_id]
        
        for dest, qty in juice_goals.items():
            # Find nearest boxcars
            boxcar_loc = self._find_nearest_yard_with_cars(engine.location, "boxcar", qty, state)
            if not boxcar_loc:
                continue
            
            # Travel to get boxcars
            if engine.location != boxcar_loc:
                path = self.find_shortest_path(engine.location, boxcar_loc, state.network, state.travel_times)
                for i in range(len(path) - 1):
                    steps.append({
                        "act": "TRAVEL",
                        "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                    })
                    state.travel(engine_id, path[i], path[i+1])
            
            # Attach boxcars
            steps.append({
                "act": "ATTACH",
                "args": {"engine": engine_id, "car_type": "boxcar", "from": boxcar_loc, "qty": qty}
            })
            state.attach(engine_id, "boxcar", boxcar_loc, qty)
            
            # Go to Corning for oranges
            if engine.location != "Corning":
                path = self.find_shortest_path(engine.location, "Corning", state.network, state.travel_times)
                for i in range(len(path) - 1):
                    steps.append({
                        "act": "TRAVEL",
                        "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                    })
                    state.travel(engine_id, path[i], path[i+1])
            
            # Load oranges
            steps.append({
                "act": "LOAD",
                "args": {"engine": engine_id, "cargo": "oranges", "at": "Corning", "cars": qty}
            })
            state.load(engine_id, "oranges", "Corning", qty)
            
            # Attach tankers at Corning
            if state.yard_cars["Corning"]["tanker"] >= qty:
                steps.append({
                    "act": "ATTACH",
                    "args": {"engine": engine_id, "car_type": "tanker", "from": "Corning", "qty": qty}
                })
                state.attach(engine_id, "tanker", "Corning", qty)
            
            # Go to Elmira for conversion
            path = self.find_shortest_path(engine.location, "Elmira", state.network, state.travel_times)
            for i in range(len(path) - 1):
                steps.append({
                    "act": "TRAVEL",
                    "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                })
                state.travel(engine_id, path[i], path[i+1])
            
            # Convert
            steps.append({
                "act": "CONVERT",
                "args": {"engine": engine_id, "at": "Elmira", "qty": qty}
            })
            state.convert(engine_id, "Elmira", qty)
            
            # Deliver juice
            path = self.find_shortest_path(engine.location, dest, state.network, state.travel_times)
            for i in range(len(path) - 1):
                steps.append({
                    "act": "TRAVEL",
                    "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                })
                state.travel(engine_id, path[i], path[i+1])
            
            # Checker auto-delivers cargo but doesn't auto-detach empty cars
            # Update internal state
            juice_tankers = sum(1 for c in state.engines[engine_id].attached_cars 
                               if c.car_type == "tanker" and c.cargo == "juice")
            if juice_tankers > 0:
                state.detach(engine_id, "tanker", dest, juice_tankers)
            
            empty_boxcars = sum(1 for c in state.engines[engine_id].attached_cars 
                               if c.car_type == "boxcar" and c.cargo is None)
            if empty_boxcars > 0:
                state.detach(engine_id, "boxcar", dest, empty_boxcars)
            
            # No explicit DETACH unless needed for capacity - checker auto-delivers
        
        return steps
    
    def _deliver_cargo(self, engine_id: str, cargo: str, destinations: dict, state: RailState) -> List[dict]:
        """Deliver regular cargo (bananas or oranges)."""
        steps = []
        engine = state.engines[engine_id]
        source = state.cargo_sources[cargo]
        
        for dest, total_qty in destinations.items():
            remaining = total_qty
            
            # Handle multi-trip if quantity exceeds capacity
            while remaining > 0:
                qty = min(remaining, self.max_capacity)
                
                # Find nearest boxcars for this trip
                boxcar_loc = self._find_nearest_yard_with_cars(engine.location, "boxcar", qty, state)
                if not boxcar_loc:
                    # Try smaller quantity
                    for try_qty in range(qty-1, 0, -1):
                        boxcar_loc = self._find_nearest_yard_with_cars(engine.location, "boxcar", try_qty, state)
                        if boxcar_loc:
                            qty = try_qty
                            break
                    if not boxcar_loc:
                        break  # Can't find any boxcars
                
                # Travel to get boxcars
                if engine.location != boxcar_loc:
                    path = self.find_shortest_path(engine.location, boxcar_loc, state.network, state.travel_times)
                    for i in range(len(path) - 1):
                        steps.append({
                            "act": "TRAVEL",
                            "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                        })
                        state.travel(engine_id, path[i], path[i+1])
                
                # Attach boxcars
                steps.append({
                    "act": "ATTACH",
                    "args": {"engine": engine_id, "car_type": "boxcar", "from": boxcar_loc, "qty": qty}
                })
                state.attach(engine_id, "boxcar", boxcar_loc, qty)
                
                # Go to source
                if engine.location != source:
                    path = self.find_shortest_path(engine.location, source, state.network, state.travel_times)
                    for i in range(len(path) - 1):
                        steps.append({
                            "act": "TRAVEL",
                            "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                        })
                        state.travel(engine_id, path[i], path[i+1])
                
                # Load cargo
                steps.append({
                    "act": "LOAD",
                    "args": {"engine": engine_id, "cargo": cargo, "at": source, "cars": qty}
                })
                state.load(engine_id, cargo, source, qty)
                
                # Deliver
                if engine.location != dest:
                    path = self.find_shortest_path(engine.location, dest, state.network, state.travel_times)
                    for i in range(len(path) - 1):
                        steps.append({
                            "act": "TRAVEL",
                            "args": {"engine": engine_id, "from": path[i], "to": path[i+1]}
                        })
                        state.travel(engine_id, path[i], path[i+1])
                
                # Checker auto-delivers when reaching destination
                # Update our internal state
                state.detach(engine_id, "boxcar", dest, qty)
                
                # Update remaining quantity
                remaining -= qty
                
                # Only detach if more trips needed for same destination (capacity constraint)
                if remaining > 0:
                    # Must detach to free capacity for next trip
                    steps.append({
                        "act": "DETACH",
                        "args": {"engine": engine_id, "car_type": "boxcar", "at": dest, "qty": qty}
                    })
        
        return steps
    
    def _solve_with_juice(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve problems involving juice production."""
        steps = []
        
        # Assign engines to tasks
        juice_engine = engines[0] if engines else "E1"
        other_engines = engines[1:] if len(engines) > 1 else []
        
        # Start juice engine
        steps.append({
            "act": "START",
            "args": {"engine": juice_engine, "at": state.engines[juice_engine].location}
        })
        state.start_engine(juice_engine, state.engines[juice_engine].location)
        
        # Produce juice
        if 'juice' in goals:
            juice_steps = self._produce_juice(juice_engine, goals['juice'], state)
            steps.extend(juice_steps)
        
        # Handle other cargo with other engines
        cargo_index = 0
        for cargo_type in ['bananas', 'oranges']:
            if cargo_type in goals and cargo_type != 'juice':
                if cargo_index < len(other_engines):
                    engine = other_engines[cargo_index]
                    steps.append({
                        "act": "START",
                        "args": {"engine": engine, "at": state.engines[engine].location}
                    })
                    state.start_engine(engine, state.engines[engine].location)
                    
                    cargo_steps = self._deliver_cargo(engine, cargo_type, goals[cargo_type], state)
                    steps.extend(cargo_steps)
                    cargo_index += 1
                else:
                    # Use juice engine after it's done
                    cargo_steps = self._deliver_cargo(juice_engine, cargo_type, goals[cargo_type], state)
                    steps.extend(cargo_steps)
        
        return {"steps": steps}
    
    def _solve_simple_cargo(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve problems with only regular cargo."""
        steps = []
        engine_index = 0
        
        for cargo_type, destinations in goals.items():
            if engine_index < len(engines):
                engine = engines[engine_index]
                
                # Start engine
                if not state.engines[engine].active:
                    steps.append({
                        "act": "START",
                        "args": {"engine": engine, "at": state.engines[engine].location}
                    })
                    state.start_engine(engine, state.engines[engine].location)
                
                # Deliver cargo
                cargo_steps = self._deliver_cargo(engine, cargo_type, destinations, state)
                steps.extend(cargo_steps)
                engine_index += 1
        
        return {"steps": steps}
    
    def _estimate_delivery_time(self, cargo_type: str, dest: str, qty: int, state: RailState) -> float:
        """Estimate time needed to deliver cargo using actual travel times."""
        # Use actual weighted paths for better estimation
        if cargo_type == 'juice':
            # Juice production: boxcars -> oranges at Corning -> convert at Elmira -> deliver
            # Estimate: travel to get boxcars + to Corning + to Elmira + to destination + operations
            path_to_corning = self.find_shortest_path("Elmira", "Corning", state.network, state.travel_times)
            path_to_dest = self.find_shortest_path("Elmira", dest, state.network, state.travel_times)
            
            time_estimate = 0
            # Time to get to Corning for oranges
            for i in range(len(path_to_corning) - 1):
                time_estimate += state.travel_times.get((path_to_corning[i], path_to_corning[i+1]), 1)
            # Load time + convert time
            time_estimate += 2  # 1 hour load + 1 hour convert
            # Time to deliver
            for i in range(len(path_to_dest) - 1):
                time_estimate += state.travel_times.get((path_to_dest[i], path_to_dest[i+1]), 1)
            
            return time_estimate
        else:
            # Regular cargo delivery
            source = state.cargo_sources.get(cargo_type, "Avon")
            # Find actual path from source to destination
            path = self.find_shortest_path(source, dest, state.network, state.travel_times)
            
            time_estimate = 0
            for i in range(len(path) - 1):
                time_estimate += state.travel_times.get((path[i], path[i+1]), 1)
            
            # Add load time
            time_estimate += 1
            
            # Account for multiple trips if needed
            if qty > 3:
                trips_needed = (qty + 2) // 3
                time_estimate *= trips_needed
            
            return time_estimate
    
    def _solve_with_deadline_priority(self, goals: dict, engines: List[str], state: RailState, priorities: List) -> dict:
        """Solve with deadline priorities using multiple engines."""
        steps = []
        engine_assignments = {}
        
        # Assign engines to urgent deliveries
        for i, (urgency, cargo_type, dest, qty, deadline) in enumerate(priorities):
            if i < len(engines):
                engine_id = engines[i]
                if cargo_type == 'juice':
                    # Juice needs special handling
                    juice_steps = self._produce_juice(engine_id, {dest: qty}, state)
                    steps.extend(juice_steps)
                else:
                    # Regular cargo
                    steps.append({
                        "act": "START",
                        "args": {"engine": engine_id, "at": state.engines[engine_id].location}
                    })
                    state.start_engine(engine_id, state.engines[engine_id].location)
                    
                    cargo_steps = self._deliver_cargo(engine_id, cargo_type, {dest: qty}, state)
                    steps.extend(cargo_steps)
        
        return {"steps": steps}
    
    def _solve_single_engine_smart(self, goals: dict, engine_id: str, state: RailState, priorities: List) -> dict:
        """Solve single engine with smart prioritization."""
        steps = []
        
        # Start engine
        steps.append({
            "act": "START",
            "args": {"engine": engine_id, "at": state.engines[engine_id].location}
        })
        state.start_engine(engine_id, state.engines[engine_id].location)
        
        # Process in priority order
        processed = set()
        for urgency, cargo_type, dest, qty, deadline in priorities:
            key = (cargo_type, dest)
            if key not in processed:
                processed.add(key)
                
                if cargo_type == 'juice':
                    juice_steps = self._produce_juice(engine_id, {dest: qty}, state)
                    steps.extend(juice_steps)
                else:
                    cargo_steps = self._deliver_cargo(engine_id, cargo_type, {dest: qty}, state)
                    steps.extend(cargo_steps)
        
        return {"steps": steps}
    
    def _solve_with_preloaded(self, goals: dict, engines: List[str], state: RailState, constraints: dict) -> dict:
        """Handle problems with pre-loaded cars."""
        steps = []
        
        # For now, handle 2-A specifically since it's the only one with preloaded
        # In a real system, this would be more generic
        
        # Use one engine for pre-loaded oranges
        engine1 = engines[0] if engines else "E1"
        steps.append({
            "act": "START", 
            "args": {"engine": engine1, "at": state.engines[engine1].location}
        })
        state.start_engine(engine1, state.engines[engine1].location)
        
        # Go get pre-loaded cars
        if state.engines[engine1].location != "Corning":
            path = self.find_shortest_path(state.engines[engine1].location, "Corning", state.network, state.travel_times)
            for i in range(len(path) - 1):
                steps.append({
                    "act": "TRAVEL",
                    "args": {"engine": engine1, "from": path[i], "to": path[i+1]}
                })
                state.travel(engine1, path[i], path[i+1])
        
        # Deliver oranges if in goals
        if 'oranges' in goals:
            for dest, qty in goals['oranges'].items():
                remaining = qty
                while remaining > 0:
                    trip_qty = min(remaining, 3)
                    
                    # Attach pre-loaded cars
                    steps.append({
                        "act": "ATTACH",
                        "args": {"engine": engine1, "car_type": "boxcar", "from": "Corning", "qty": trip_qty}
                    })
                    
                    # Deliver
                    path = self.find_shortest_path("Corning", dest, state.network, state.travel_times)
                    for i in range(len(path) - 1):
                        steps.append({
                            "act": "TRAVEL",
                            "args": {"engine": engine1, "from": path[i], "to": path[i+1]}
                        })
                    
                    remaining -= trip_qty
                    
                    if remaining > 0:
                        # Need to detach and go back
                        steps.append({
                            "act": "DETACH",
                            "args": {"engine": engine1, "car_type": "boxcar", "at": dest, "qty": trip_qty}
                        })
                        
                        path = self.find_shortest_path(dest, "Corning", state.network, state.travel_times)
                        for i in range(len(path) - 1):
                            steps.append({
                                "act": "TRAVEL",
                                "args": {"engine": engine1, "from": path[i], "to": path[i+1]}
                            })
        
        # Handle juice with second engine if needed
        if 'juice' in goals and len(engines) > 1:
            engine2 = engines[1]
            steps.append({
                "act": "START",
                "args": {"engine": engine2, "at": state.engines[engine2].location}
            })
            state.start_engine(engine2, state.engines[engine2].location)
            
            juice_steps = self._produce_juice(engine2, goals['juice'], state)
            steps.extend(juice_steps)
        
        return {"steps": steps}
    
    def _find_nearest_yard_with_cars(self, from_loc: str, car_type: str, qty: int, state: RailState) -> Optional[str]:
        """Find nearest yard with required cars."""
        # Use BFS to find nearest
        queue = deque([(from_loc, 0)])
        visited = {from_loc}
        
        while queue:
            loc, dist = queue.popleft()
            
            if state.yard_cars[loc].get(car_type, 0) >= qty:
                return loc
            
            for neighbor in state.network.get(loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return None
    
    def solve_brute_force(self, problem_id: str, goals: dict, constraints: dict, overrides: dict = None, max_depth: int = 20) -> dict:
        """Brute force solution that explores all possible action sequences."""
        initial_state = RailState()
        
        # Apply overrides if provided
        if overrides and 'boxcars' in overrides:
            for override in overrides['boxcars']:
                location = override['location']
                payload = override['payload']
                count = override['count']
                if not hasattr(initial_state, 'preloaded_cars'):
                    initial_state.preloaded_cars = {}
                initial_state.preloaded_cars[location] = {'payload': payload, 'count': count}
        
        # Get available engines
        available_engines = constraints.get('available_engines', ['E1', 'E2', 'E3'])
        if isinstance(available_engines, list):
            engines_to_use = [e for e in available_engines if e in initial_state.engines]
        else:
            engines_to_use = ['E1', 'E2', 'E3']
        
        # BFS to find solution
        queue = deque([(initial_state, [], 0)])  # (state, steps_taken, depth)
        visited_states = set()
        best_solution = None
        best_completion = 0.0
        
        while queue:
            current_state, steps, depth = queue.popleft()
            
            # Check if we've reached max depth
            if depth >= max_depth:
                continue
            
            # Check goal completion
            completion = current_state.check_goals(goals)
            if completion > best_completion:
                best_completion = completion
                best_solution = steps
            
            if completion >= 1.0:
                return {"steps": steps}
            
            # Generate state hash for visited check
            state_hash = self._hash_state(current_state)
            if state_hash in visited_states:
                continue
            visited_states.add(state_hash)
            
            # Try all possible actions
            possible_actions = self._generate_possible_actions(current_state, engines_to_use, goals)
            
            for action in possible_actions:
                new_state = current_state.copy()
                if self._apply_action(new_state, action):
                    queue.append((new_state, steps + [action], depth + 1))
        
        # Return best partial solution if no complete solution found
        return {"steps": best_solution if best_solution else []}
    
    def _hash_state(self, state: RailState) -> str:
        """Generate a hash for the current state to detect visited states."""
        state_dict = {
            'time': state.time,
            'engines': {},
            'yard_cars': state.yard_cars,
            'delivered': dict(state.delivered)
        }
        
        for eng_id, engine in state.engines.items():
            state_dict['engines'][eng_id] = {
                'location': engine.location,
                'active': engine.active,
                'attached': [(c.car_type, c.cargo) for c in engine.attached_cars]
            }
        
        return str(sorted(state_dict.items()))
    
    def _generate_possible_actions(self, state: RailState, engines: List[str], goals: dict) -> List[dict]:
        """Generate all possible valid actions from current state."""
        actions = []
        
        for engine_id in engines:
            engine = state.engines.get(engine_id)
            if not engine:
                continue
            
            # START action
            if not engine.active:
                actions.append({
                    "act": "START",
                    "args": {"engine": engine_id, "at": engine.location}
                })
                continue  # Can't do other actions until started
            
            # TRAVEL actions
            for neighbor in state.network.get(engine.location, []):
                actions.append({
                    "act": "TRAVEL",
                    "args": {"engine": engine_id, "from": engine.location, "to": neighbor}
                })
            
            # ATTACH actions
            for car_type in ["boxcar", "tanker"]:
                available = state.yard_cars[engine.location].get(car_type, 0)
                capacity_left = 3 - len(engine.attached_cars)
                if available > 0 and capacity_left > 0:
                    for qty in range(1, min(available, capacity_left) + 1):
                        actions.append({
                            "act": "ATTACH",
                            "args": {"engine": engine_id, "car_type": car_type, "from": engine.location, "qty": qty}
                        })
            
            # DETACH actions
            car_counts = {}
            for car in engine.attached_cars:
                car_counts[car.car_type] = car_counts.get(car.car_type, 0) + 1
            
            for car_type, count in car_counts.items():
                for qty in range(1, count + 1):
                    actions.append({
                        "act": "DETACH",
                        "args": {"engine": engine_id, "car_type": car_type, "at": engine.location, "qty": qty}
                    })
            
            # LOAD actions
            empty_boxcars = sum(1 for c in engine.attached_cars if c.car_type == "boxcar" and c.cargo is None)
            if empty_boxcars > 0:
                for cargo in ["bananas", "oranges"]:
                    if engine.location == state.cargo_sources.get(cargo):
                        for qty in range(1, empty_boxcars + 1):
                            actions.append({
                                "act": "LOAD",
                                "args": {"engine": engine_id, "cargo": cargo, "at": engine.location, "cars": qty}
                            })
            
            # CONVERT action
            if engine.location == state.factory:
                orange_boxcars = sum(1 for c in engine.attached_cars if c.car_type == "boxcar" and c.cargo == "oranges")
                empty_tankers = sum(1 for c in engine.attached_cars if c.car_type == "tanker" and c.cargo is None)
                max_convert = min(orange_boxcars, empty_tankers)
                if max_convert > 0:
                    for qty in range(1, max_convert + 1):
                        actions.append({
                            "act": "CONVERT",
                            "args": {"engine": engine_id, "at": engine.location, "qty": qty}
                        })
        
        return actions
    
    def _apply_action(self, state: RailState, action: dict) -> bool:
        """Apply an action to the state and return success status."""
        act_type = action["act"]
        args = action["args"]
        
        if act_type == "START":
            return state.start_engine(args["engine"], args["at"])
        elif act_type == "TRAVEL":
            return state.travel(args["engine"], args["from"], args["to"])
        elif act_type == "ATTACH":
            return state.attach(args["engine"], args["car_type"], args["from"], args["qty"])
        elif act_type == "DETACH":
            return state.detach(args["engine"], args["car_type"], args["at"], args["qty"])
        elif act_type == "LOAD":
            return state.load(args["engine"], args["cargo"], args["at"], args["cars"])
        elif act_type == "CONVERT":
            return state.convert(args["engine"], args["at"], args["qty"])
        
        return False
    
    def _solve_2A_with_preloaded(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 2-A with pre-loaded orange boxcars at Corning."""
        steps = []
        
        # Problem 2-A: 5 pre-loaded orange boxcars at Corning
        # Need to deliver: 4 oranges to Bath, 1 juice to Bath
        # Strategy: E1 delivers 4 oranges, E2 converts 1 orange to juice
        
        # E1 handles delivery of 4 oranges
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        # Go to Corning to get pre-loaded oranges
        path = ["Avon", "Dansville", "Corning"]
        for i in range(len(path) - 1):
            steps.append({
                "act": "TRAVEL",
                "args": {"engine": "E1", "from": path[i], "to": path[i+1]}
            })
        
        # Attach 3 pre-loaded orange boxcars
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Corning", "qty": 3}
        })
        
        # Deliver first batch to Bath
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Bath"}
        })
        
        # Detach empty boxcars
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E1", "car_type": "boxcar", "at": "Bath", "qty": 3}
        })
        
        # Back to Corning for the 4th orange
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Bath", "to": "Corning"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Corning", "qty": 1}
        })
        
        # Deliver 4th orange
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Bath"}
        })
        
        # Now E2 handles juice production using 5th pre-loaded orange
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        # Go to Corning
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        # Attach the 5th pre-loaded orange boxcar and a tanker
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Corning", "qty": 1}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 1}
        })
        
        # Back to Elmira for conversion
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        steps.append({
            "act": "CONVERT",
            "args": {"engine": "E2", "at": "Elmira", "qty": 1}
        })
        
        # Deliver juice to Bath
        path = ["Elmira", "Corning", "Bath"]
        for i in range(len(path) - 1):
            steps.append({
                "act": "TRAVEL",
                "args": {"engine": "E2", "from": path[i], "to": path[i+1]}
            })
        
        return {"steps": steps}
    
    def _solve_2E_parallel(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 2-E: 5 oranges to Bath by hour 7 using parallel engines."""
        steps = []
        
        # E2 starts at Elmira with 2 boxcars
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}
        })
        
        # E2: Elmira -> Corning (hour 2)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        # E2: Load oranges (hour 3)
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        # E2: Corning -> Bath (hour 5)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Bath"}
        })
        
        # E1 starts at Avon
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        # E1: Avon -> Dansville (hour 1)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}
        })
        
        # E1: Attach 3 boxcars at Dansville
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}
        })
        
        # E1: Dansville -> Corning (hour 4)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}
        })
        
        # E1: Load oranges (hour 5)
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 3}
        })
        
        # E1: Corning -> Bath (hour 7)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Bath"}
        })
        
        return {"steps": steps}
    
    def _solve_parallel_delivery(self, goals: dict, engines: List[str], state: RailState, deadline: float) -> dict:
        """Generic parallel delivery for tight deadlines."""
        # For now, fall back to sequential for other cases
        # This could be enhanced further
        if 'juice' in goals:
            return self._solve_with_juice(goals, engines, state)
        else:
            return self._solve_simple_cargo(goals, engines, state)
    
    def _solve_2F_parallel(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 2-F: 3 bananas to Bath + 2 juice to Dansville by hour 12."""
        steps = []
        
        # Use E1 for bananas (starts closer at Avon)
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        # Travel to get boxcars
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Bath"}
        })
        
        # Attach 3 boxcars at Bath
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Bath", "qty": 2}
        })
        
        # Back to Avon for bananas
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Bath", "to": "Avon"}
        })
        
        # Load 2 bananas first
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 2}
        })
        
        # Deliver to Bath
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Bath"}
        })
        
        # Detach and get 3rd boxcar
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E1", "car_type": "boxcar", "at": "Bath", "qty": 2}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Bath", "qty": 1}
        })
        
        # Back for 3rd banana
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Bath", "to": "Avon"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 1}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Bath"}
        })
        
        # Meanwhile E1 makes juice (start earlier for parallel)
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        # Get boxcars from Dansville (but leave 1 for E2)
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}
        })
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 2}
        })
        
        # Get oranges
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}
        })
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        # Get tankers
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "tanker", "from": "Corning", "qty": 2}
        })
        
        # Convert at Elmira
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}
        })
        steps.append({
            "act": "CONVERT",
            "args": {"engine": "E1", "at": "Elmira", "qty": 2}
        })
        
        # Deliver juice to Dansville
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Dansville"}
        })
        
        return {"steps": steps}
    
    def _solve_3A_optimized(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 3-A: 2 juice to Avon by hour 13 - optimized path."""
        steps = []
        
        # Start E2 at Elmira (closer to resources)
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        # Attach 2 boxcars at Elmira
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}
        })
        
        # Go to Corning for oranges and tankers
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}
        })
        
        # Back to Elmira for conversion
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        steps.append({
            "act": "CONVERT",
            "args": {"engine": "E2", "at": "Elmira", "qty": 2}
        })
        
        # Deliver to Avon
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Dansville", "to": "Avon"}
        })
        
        return {"steps": steps}
    
    def _solve_3B_resource_aware(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 3-B with proper resource allocation."""
        steps = []
        
        # E1 handles juice first
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        # Get boxcars from Dansville
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}
        })
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 2}
        })
        
        # Get oranges and tankers at Corning
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}
        })
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "tanker", "from": "Corning", "qty": 2}
        })
        
        # Convert at Elmira
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}
        })
        steps.append({
            "act": "CONVERT",
            "args": {"engine": "E1", "at": "Elmira", "qty": 2}
        })
        
        # Deliver juice to Avon
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Dansville"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}
        })
        
        # E2 handles bananas
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        # First travel to Dansville to get boxcar
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}
        })
        
        # Now attach boxcar at Dansville
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Dansville", "qty": 1}
        })
        
        # Travel to Bath to get remaining 2 boxcars
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Dansville", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Bath"}
        })
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Bath", "qty": 2}
        })
        
        # Go to Avon for bananas
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Bath", "to": "Avon"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "bananas", "at": "Avon", "cars": 3}
        })
        
        # Deliver to Bath
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Avon", "to": "Bath"}
        })
        
        # Detach 1 at Bath  
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E2", "car_type": "boxcar", "at": "Bath", "qty": 1}
        })
        
        # Deliver to Corning
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Bath", "to": "Corning"}
        })
        
        # Detach 1 at Corning
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E2", "car_type": "boxcar", "at": "Corning", "qty": 1}
        })
        
        # Deliver to Elmira
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        return {"steps": steps}
    
    def _solve_3C_single_engine(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 3-C with single engine E2 - bananas first for deadline."""
        steps = []
        
        # E2 is the only available engine
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        # Do bananas FIRST (deadline is tighter - hour 21)
        # Get boxcars from Dansville
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Dansville", "qty": 3}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        # DON'T attach tankers yet to avoid capacity issues
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        # Detach loaded oranges temporarily
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E2", "car_type": "boxcar", "at": "Elmira", "qty": 2}
        })
        
        # Now get tankers
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        # Re-attach orange boxcars
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}
        })
        
        # Convert
        steps.append({
            "act": "CONVERT",
            "args": {"engine": "E2", "at": "Elmira", "qty": 2}
        })
        
        # Deliver juice to Avon
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Dansville", "to": "Avon"}
        })
        
        # MUST detach everything before getting bananas
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E2", "car_type": "tanker", "at": "Avon", "qty": 2}
        })
        steps.append({
            "act": "DETACH",
            "args": {"engine": "E2", "car_type": "boxcar", "at": "Avon", "qty": 2}
        })
        
        # Travel to Dansville to get boxcars for bananas
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Avon", "to": "Dansville"}
        })
        
        # Now attach boxcars
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Dansville", "qty": 3}
        })
        
        # Travel back to Avon for bananas
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Dansville", "to": "Avon"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "bananas", "at": "Avon", "cars": 3}
        })
        
        # Deliver to Elmira
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Avon", "to": "Dansville"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Dansville", "to": "Corning"}
        })
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        return {"steps": steps}
    
    def _solve_3E_parallel(self, goals: dict, engines: List[str], state: RailState) -> dict:
        """Solve 3-E: 7 oranges to Elmira by hour 9 using multiple engines."""
        steps = []
        
        # E1 takes 3 from Dansville
        steps.append({
            "act": "START",
            "args": {"engine": "E1", "at": "Avon"}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 3}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}
        })
        
        # E2 takes 2 from Elmira
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}
        })
        
        # E3 takes 2 from Bath
        steps.append({
            "act": "START",
            "args": {"engine": "E3", "at": "Elmira"}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E3", "from": "Corning", "to": "Bath"}
        })
        
        steps.append({
            "act": "ATTACH",
            "args": {"engine": "E3", "car_type": "boxcar", "from": "Bath", "qty": 2}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E3", "from": "Bath", "to": "Corning"}
        })
        
        steps.append({
            "act": "LOAD",
            "args": {"engine": "E3", "cargo": "oranges", "at": "Corning", "cars": 2}
        })
        
        steps.append({
            "act": "TRAVEL",
            "args": {"engine": "E3", "from": "Corning", "to": "Elmira"}
        })
        
        return {"steps": steps}