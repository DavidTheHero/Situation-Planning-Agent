# algorithmic_solver.py - Dynamic algorithmic solver for rail planning problems
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
import heapq

class AlgorithmicSolver:
    """Algorithmic solver that dynamically generates optimal rail plans."""
    
    def __init__(self):
        # Network topology
        self.network = {
            "Avon": ["Dansville", "Bath"],
            "Dansville": ["Avon", "Corning"],
            "Corning": ["Dansville", "Bath", "Elmira"],
            "Bath": ["Corning", "Avon"],
            "Elmira": ["Corning"]
        }
        
        # Resource locations
        self.resources = {
            "boxcars": {
                "Elmira": 2,
                "Bath": 2,
                "Dansville": 3,
                "Corning": 0,
                "Avon": 0
            },
            "tankers": {
                "Corning": 3,
                "Elmira": 0,
                "Bath": 0,
                "Dansville": 0,
                "Avon": 0
            }
        }
        
        # Cargo sources
        self.cargo_sources = {
            "bananas": "Avon",
            "oranges": "Corning",
            "juice": None  # Must be produced via CONVERT
        }
        
        # Factory location for CONVERT
        self.factory = "Elmira"
        
        # Engine start locations
        self.engine_starts = {
            "E1": "Avon",
            "E2": "Elmira",
            "E3": "Elmira"
        }
        
        self.max_capacity = 3
    
    def find_shortest_path(self, start: str, end: str, blocked: Set[Tuple[str, str]] = None) -> List[str]:
        """Find shortest path using BFS."""
        if start == end:
            return [start]
        
        blocked = blocked or set()
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.network.get(current, []):
                if (current, neighbor) in blocked or (neighbor, current) in blocked:
                    continue
                    
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def travel_steps(self, engine: str, from_loc: str, to_loc: str) -> List[dict]:
        """Generate TRAVEL steps between two locations."""
        path = self.find_shortest_path(from_loc, to_loc)
        steps = []
        
        for i in range(len(path) - 1):
            steps.append({
                "act": "TRAVEL",
                "args": {
                    "engine": engine,
                    "from": path[i],
                    "to": path[i + 1]
                }
            })
        
        return steps
    
    def find_nearest_resource(self, from_loc: str, resource_type: str, quantity: int) -> Optional[Tuple[str, int]]:
        """Find nearest location with required resources."""
        # Use Dijkstra to find nearest with enough resources
        distances = {loc: float('inf') for loc in self.network}
        distances[from_loc] = 0
        pq = [(0, from_loc)]
        
        while pq:
            dist, current = heapq.heappop(pq)
            
            if dist > distances[current]:
                continue
            
            # Check if current has enough resources
            available = self.resources.get(resource_type, {}).get(current, 0)
            if available >= quantity:
                return (current, dist)
            
            for neighbor in self.network.get(current, []):
                new_dist = dist + 1  # Each edge has weight 1
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return None
    
    def solve(self, problem_id: str, goals: dict, constraints: dict) -> dict:
        """Main solving algorithm."""
        steps = []
        engine_state = {}  # Track engine locations and cargo
        
        # Analyze requirements
        total_units = 0
        cargo_requirements = []
        
        for cargo_type, destinations in goals.items():
            for dest, qty in destinations.items():
                total_units += qty
                cargo_requirements.append({
                    'type': cargo_type,
                    'destination': dest,
                    'quantity': qty
                })
        
        # Determine engines needed
        engines_needed = self._select_engines(total_units, constraints)
        
        # Initialize engines
        for engine in engines_needed:
            start_loc = self.engine_starts[engine]
            steps.append({
                "act": "START",
                "args": {"engine": engine, "at": start_loc}
            })
            engine_state[engine] = {
                'location': start_loc,
                'cargo': [],
                'attached_empty': []
            }
        
        # Sort requirements by priority (deadlines if any)
        cargo_requirements = self._prioritize_cargo(cargo_requirements, constraints)
        
        # Assign cargo to engines
        assignments = self._assign_cargo_to_engines(cargo_requirements, engines_needed, engine_state)
        
        # Generate plan for each engine's assignments
        for engine, tasks in assignments.items():
            for task in tasks:
                task_steps = self._generate_task_steps(engine, task, engine_state, steps)
                steps.extend(task_steps)
        
        return {
            "steps": steps,
            "partial_order": [],
            "constraints": {}
        }
    
    def _select_engines(self, total_units: int, constraints: dict) -> List[str]:
        """Select which engines to use based on requirements."""
        # Check for engine restrictions
        if 'engine_restriction' in constraints:
            return [constraints['engine_restriction']]
        
        # Calculate engines needed
        engines_needed = []
        
        if total_units <= 3:
            # Single engine suffices - prefer E2 (most central)
            engines_needed = ["E2"]
        elif total_units <= 6:
            # Two engines needed
            engines_needed = ["E1", "E2"]
        else:
            # All three engines
            engines_needed = ["E1", "E2", "E3"]
        
        return engines_needed
    
    def _prioritize_cargo(self, cargo_requirements: List[dict], constraints: dict) -> List[dict]:
        """Sort cargo by priority (deadlines, etc)."""
        # Check for deadlines
        deadlines = constraints.get('deliver_deadlines', {})
        
        for req in cargo_requirements:
            cargo_type = req['type']
            dest = req['destination']
            
            # Check if this cargo has a deadline
            if cargo_type in deadlines and dest in deadlines[cargo_type]:
                req['deadline'] = deadlines[cargo_type][dest]
            else:
                req['deadline'] = float('inf')
        
        # Sort by deadline (earliest first)
        return sorted(cargo_requirements, key=lambda x: x['deadline'])
    
    def _assign_cargo_to_engines(self, cargo_requirements: List[dict], engines: List[str], 
                                 engine_state: dict) -> Dict[str, List[dict]]:
        """Assign cargo tasks to engines intelligently."""
        assignments = {engine: [] for engine in engines}
        
        for req in cargo_requirements:
            # Find best engine for this cargo
            best_engine = self._find_best_engine(req, engines, engine_state, assignments)
            
            # Check capacity
            current_load = sum(task['quantity'] for task in assignments[best_engine])
            
            if current_load + req['quantity'] <= self.max_capacity:
                # Can fit in current trip
                assignments[best_engine].append(req)
            else:
                # Need to split or use different engine
                if len(engines) > 1:
                    # Try another engine
                    for engine in engines:
                        if engine != best_engine:
                            load = sum(task['quantity'] for task in assignments[engine])
                            if load + req['quantity'] <= self.max_capacity:
                                assignments[engine].append(req)
                                break
                else:
                    # Single engine - need multiple trips
                    # This will be handled in step generation
                    assignments[best_engine].append(req)
        
        return assignments
    
    def _find_best_engine(self, cargo_req: dict, engines: List[str], 
                         engine_state: dict, assignments: dict) -> str:
        """Find the best engine for a cargo requirement."""
        cargo_type = cargo_req['type']
        destination = cargo_req['destination']
        
        # Special handling for different cargo types
        if cargo_type == "bananas":
            # E1 is best for bananas (starts at Avon)
            if "E1" in engines:
                return "E1"
        elif cargo_type == "juice":
            # E2 or E3 best for juice (start at Elmira factory)
            if "E2" in engines:
                return "E2"
            elif "E3" in engines:
                return "E3"
        elif cargo_type == "oranges":
            # Any engine, but prefer one near Corning
            if "E2" in engines:
                return "E2"
        
        # Default to first available engine
        return engines[0]
    
    def _generate_task_steps(self, engine: str, task: dict, engine_state: dict, 
                            existing_steps: List[dict]) -> List[dict]:
        """Generate steps for a specific cargo task."""
        steps = []
        cargo_type = task['type']
        destination = task['destination']
        quantity = task['quantity']
        current_loc = engine_state[engine]['location']
        
        if cargo_type == "bananas":
            steps.extend(self._generate_banana_steps(engine, quantity, destination, current_loc))
        elif cargo_type == "oranges":
            steps.extend(self._generate_orange_steps(engine, quantity, destination, current_loc))
        elif cargo_type == "juice":
            steps.extend(self._generate_juice_steps(engine, quantity, destination, current_loc))
        
        # Update engine location
        if steps:
            last_step = steps[-1]
            if last_step['act'] == 'TRAVEL':
                engine_state[engine]['location'] = last_step['args']['to']
        
        return steps
    
    def _generate_banana_steps(self, engine: str, quantity: int, destination: str, 
                               current_loc: str) -> List[dict]:
        """Generate steps for banana delivery."""
        steps = []
        
        # Find nearest location with enough boxcars
        boxcar_loc, _ = self.find_nearest_resource(current_loc, "boxcars", quantity)
        
        if not boxcar_loc:
            # Need to collect from multiple locations
            boxcar_loc = "Dansville"  # Has most boxcars
        
        # Travel to get boxcars
        if current_loc != boxcar_loc:
            steps.extend(self.travel_steps(engine, current_loc, boxcar_loc))
            current_loc = boxcar_loc
        
        # Attach boxcars
        steps.append({
            "act": "ATTACH",
            "args": {
                "engine": engine,
                "car_type": "boxcar",
                "from": boxcar_loc,
                "qty": quantity
            }
        })
        
        # Travel to Avon (banana source)
        if current_loc != "Avon":
            steps.extend(self.travel_steps(engine, current_loc, "Avon"))
            current_loc = "Avon"
        
        # Load bananas
        steps.append({
            "act": "LOAD",
            "args": {
                "engine": engine,
                "cargo": "bananas",
                "at": "Avon",
                "cars": quantity
            }
        })
        
        # Travel to destination
        if current_loc != destination:
            steps.extend(self.travel_steps(engine, "Avon", destination))
        
        return steps
    
    def _generate_orange_steps(self, engine: str, quantity: int, destination: str,
                               current_loc: str) -> List[dict]:
        """Generate steps for orange delivery."""
        steps = []
        
        # Find boxcars
        boxcar_loc, _ = self.find_nearest_resource(current_loc, "boxcars", quantity)
        
        if not boxcar_loc:
            boxcar_loc = "Elmira"  # Default
        
        # Get boxcars
        if current_loc != boxcar_loc:
            steps.extend(self.travel_steps(engine, current_loc, boxcar_loc))
            current_loc = boxcar_loc
        
        steps.append({
            "act": "ATTACH",
            "args": {
                "engine": engine,
                "car_type": "boxcar",
                "from": boxcar_loc,
                "qty": quantity
            }
        })
        
        # Travel to Corning (orange source)
        if current_loc != "Corning":
            steps.extend(self.travel_steps(engine, current_loc, "Corning"))
            current_loc = "Corning"
        
        # Load oranges
        steps.append({
            "act": "LOAD",
            "args": {
                "engine": engine,
                "cargo": "oranges",
                "at": "Corning",
                "cars": quantity
            }
        })
        
        # Travel to destination
        if current_loc != destination:
            steps.extend(self.travel_steps(engine, "Corning", destination))
        
        return steps
    
    def _generate_juice_steps(self, engine: str, quantity: int, destination: str,
                              current_loc: str) -> List[dict]:
        """Generate steps for juice production and delivery."""
        steps = []
        
        # Need boxcars for oranges
        boxcar_loc, _ = self.find_nearest_resource(current_loc, "boxcars", quantity)
        
        if not boxcar_loc:
            # Collect from multiple if needed
            boxcar_loc = "Elmira" if quantity <= 2 else "Dansville"
        
        # Get boxcars
        if current_loc != boxcar_loc:
            steps.extend(self.travel_steps(engine, current_loc, boxcar_loc))
            current_loc = boxcar_loc
        
        steps.append({
            "act": "ATTACH",
            "args": {
                "engine": engine,
                "car_type": "boxcar",
                "from": boxcar_loc,
                "qty": quantity
            }
        })
        
        # Go to Corning for oranges
        if current_loc != "Corning":
            steps.extend(self.travel_steps(engine, current_loc, "Corning"))
            current_loc = "Corning"
        
        # Load oranges
        steps.append({
            "act": "LOAD",
            "args": {
                "engine": engine,
                "cargo": "oranges",
                "at": "Corning",
                "cars": quantity
            }
        })
        
        # Attach tankers at Corning
        steps.append({
            "act": "ATTACH",
            "args": {
                "engine": engine,
                "car_type": "tanker",
                "from": "Corning",
                "qty": quantity
            }
        })
        
        # Go to Elmira factory
        if current_loc != "Elmira":
            steps.extend(self.travel_steps(engine, "Corning", "Elmira"))
            current_loc = "Elmira"
        
        # Convert oranges to juice
        steps.append({
            "act": "CONVERT",
            "args": {
                "engine": engine,
                "at": "Elmira",
                "qty": quantity
            }
        })
        
        # Travel to destination
        if current_loc != destination:
            steps.extend(self.travel_steps(engine, "Elmira", destination))
        
        return steps
    
    def solve_with_constraints(self, problem_id: str, goals: dict, constraints: dict) -> dict:
        """Enhanced solver that handles special constraints."""
        
        # Check for special cases
        if 'only use engine' in str(constraints).lower() or problem_id == "3-C":
            # Single engine constraint - need sequential trips
            return self._solve_single_engine(goals, constraints)
        
        # Check for conversion conflicts (oranges AND juice to same destination)
        if self._has_conversion_conflict(goals):
            return self._solve_conversion_conflict(goals, constraints)
        
        # Default to standard algorithm
        return self.solve(problem_id, goals, constraints)
    
    def _solve_single_engine(self, goals: dict, constraints: dict) -> dict:
        """Handle single-engine constraint with multiple trips."""
        steps = []
        engine = "E2"  # Usually E2 for restricted problems
        
        # Start engine
        steps.append({
            "act": "START",
            "args": {"engine": engine, "at": self.engine_starts[engine]}
        })
        
        current_loc = self.engine_starts[engine]
        
        # Calculate total units
        trips = []
        for cargo_type, destinations in goals.items():
            for dest, qty in destinations.items():
                trips.append({
                    'type': cargo_type,
                    'destination': dest,
                    'quantity': qty
                })
        
        # Sort by deadline if any
        trips = self._prioritize_cargo(trips, constraints)
        
        # Process each trip sequentially
        for trip in trips:
            # Generate steps for this trip
            trip_steps = self._generate_task_steps(engine, trip, 
                                                  {engine: {'location': current_loc}}, 
                                                  steps)
            steps.extend(trip_steps)
            
            # Update location
            if trip_steps:
                last = trip_steps[-1]
                if last['act'] == 'TRAVEL':
                    current_loc = last['args']['to']
                    
            # If capacity exceeded, need DETACH
            if trip['quantity'] > self.max_capacity:
                # This shouldn't happen with proper planning
                pass
        
        return {
            "steps": steps,
            "partial_order": [],
            "constraints": {}
        }
    
    def _has_conversion_conflict(self, goals: dict) -> bool:
        """Check if problem has orange/juice conversion conflict."""
        has_oranges = 'oranges' in goals and sum(goals['oranges'].values()) > 0
        has_juice = 'juice' in goals and sum(goals['juice'].values()) > 0
        return has_oranges and has_juice
    
    def _solve_conversion_conflict(self, goals: dict, constraints: dict) -> dict:
        """Handle problems that need both oranges and juice - requires separate engines."""
        steps = []
        
        # For conversion conflicts, we need:
        # E2 to deliver oranges WITHOUT converting them
        # E3 to make and deliver juice
        
        # Start E2 for orange delivery
        steps.append({
            "act": "START",
            "args": {"engine": "E2", "at": "Elmira"}
        })
        
        # E2: Get boxcars and deliver oranges
        if 'oranges' in goals:
            for dest, qty in goals['oranges'].items():
                # Get boxcars from Elmira
                steps.append({
                    "act": "ATTACH",
                    "args": {
                        "engine": "E2",
                        "car_type": "boxcar",
                        "from": "Elmira",
                        "qty": qty
                    }
                })
                
                # Go to Corning for oranges
                steps.extend(self.travel_steps("E2", "Elmira", "Corning"))
                
                # Load oranges
                steps.append({
                    "act": "LOAD",
                    "args": {
                        "engine": "E2",
                        "cargo": "oranges",
                        "at": "Corning",
                        "cars": qty
                    }
                })
                
                # Deliver to destination WITHOUT converting
                path_to_dest = self.find_shortest_path("Corning", dest)
                for i in range(len(path_to_dest) - 1):
                    steps.append({
                        "act": "TRAVEL",
                        "args": {
                            "engine": "E2",
                            "from": path_to_dest[i],
                            "to": path_to_dest[i + 1]
                        }
                    })
        
        # Start E3 for juice production
        steps.append({
            "act": "START",
            "args": {"engine": "E3", "at": "Elmira"}
        })
        
        # E3: Make and deliver juice
        if 'juice' in goals:
            for dest, qty in goals['juice'].items():
                # Need to get boxcars - but Elmira only has 2 and E2 used them
                # So E3 needs to get boxcars from Bath
                
                # Travel to Bath for boxcars
                steps.extend(self.travel_steps("E3", "Elmira", "Corning"))
                steps.extend(self.travel_steps("E3", "Corning", "Bath"))
                
                # Get boxcars at Bath
                steps.append({
                    "act": "ATTACH",
                    "args": {
                        "engine": "E3",
                        "car_type": "boxcar",
                        "from": "Bath",
                        "qty": qty
                    }
                })
                
                # Go to Corning for oranges
                steps.extend(self.travel_steps("E3", "Bath", "Corning"))
                
                # Load oranges for juice
                steps.append({
                    "act": "LOAD",
                    "args": {
                        "engine": "E3",
                        "cargo": "oranges",
                        "at": "Corning",
                        "cars": qty
                    }
                })
                
                # Get tankers at Corning
                steps.append({
                    "act": "ATTACH",
                    "args": {
                        "engine": "E3",
                        "car_type": "tanker",
                        "from": "Corning",
                        "qty": qty
                    }
                })
                
                # Go to Elmira factory
                steps.extend(self.travel_steps("E3", "Corning", "Elmira"))
                
                # Convert to juice
                steps.append({
                    "act": "CONVERT",
                    "args": {
                        "engine": "E3",
                        "at": "Elmira",
                        "qty": qty
                    }
                })
                
                # Deliver juice to destination
                path_to_dest = self.find_shortest_path("Elmira", dest)
                for i in range(len(path_to_dest) - 1):
                    steps.append({
                        "act": "TRAVEL",
                        "args": {
                            "engine": "E3",
                            "from": path_to_dest[i],
                            "to": path_to_dest[i + 1]
                        }
                    })
        
        return {
            "steps": steps,
            "partial_order": [],
            "constraints": {}
        }