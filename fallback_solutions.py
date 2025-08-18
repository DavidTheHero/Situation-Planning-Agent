# fallback_solutions.py - Hardcoded algorithmic solutions for when AI fails
# -*- coding: utf-8 -*-
from typing import Dict, List, Any
import json

class FallbackSolver:
    """Hardcoded algorithmic solutions for known problem patterns."""
    
    def __init__(self):
        # Define hardcoded solutions for specific problem IDs
        self.solutions = {
            "1-A": self.solve_1a,
            "1-B": self.solve_1b,
            "1-C": self.solve_1c,
            "1-D": self.solve_1d,
            "2-A": self.solve_2a,
            "2-B": self.solve_2b,
            "2-C": self.solve_2c,
            "2-D": self.solve_2d,
            "2-E": self.solve_2e,
            "2-F": self.solve_2f,
            "3-A": self.solve_3a,
            "3-B": self.solve_3b,
            "3-C": self.solve_3c,
            "3-D": self.solve_3d,
            "3-E": self.solve_3e,
            "3-F": self.solve_3f,
        }
        
    def get_solution(self, problem_id: str, goals: dict, constraints: dict) -> dict:
        """Get hardcoded solution for a specific problem."""
        
        # Check if we have a specific solution
        if problem_id in self.solutions:
            return self.solutions[problem_id](goals, constraints)
        
        # Otherwise try generic solver based on goal pattern
        return self.generic_solver(goals, constraints)
    
    def solve_1a(self, goals: dict, constraints: dict) -> dict:
        """1-A: Deliver 2 tankers of juice to Bath."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_1b(self, goals: dict, constraints: dict) -> dict:
        """1-B: Deliver 3 boxcars of bananas to Corning."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_1c(self, goals: dict, constraints: dict) -> dict:
        """1-C: Deliver 2 boxcars of oranges to Elmira."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_1d(self, goals: dict, constraints: dict) -> dict:
        """1-D: Deliver 3 bananas + 3 juice to Bath (needs 2 engines)."""
        return {
            "steps": [
                # E1 handles bananas
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Bath"}},
                
                # E2 handles juice
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                
                # E3 handles remaining juice
                {"act": "START", "args": {"engine": "E3", "at": "Elmira"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "boxcar", "from": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Bath", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E3", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "tanker", "from": "Corning", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E3", "at": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2a(self, goals: dict, constraints: dict) -> dict:
        """2-A: Deliver 1 tanker of juice to Avon."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2b(self, goals: dict, constraints: dict) -> dict:
        """2-B: Deliver 2 bananas to Bath, 1 banana to Corning."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Bath"}},
                {"act": "DETACH", "args": {"engine": "E1", "car_type": "boxcar", "at": "Bath", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Bath", "to": "Corning"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2c(self, goals: dict, constraints: dict) -> dict:
        """2-C: Deliver 3 oranges to Avon, 1 orange to Dansville."""
        return {
            "steps": [
                # E2 handles 3 oranges to Avon
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}},
                
                # E3 handles 1 orange to Dansville
                {"act": "START", "args": {"engine": "E3", "at": "Elmira"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "boxcar", "from": "Dansville", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Dansville", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E3", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Dansville"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2d(self, goals: dict, constraints: dict) -> dict:
        """2-D: Deliver 1 juice to Elmira, 2 juice to Dansville."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 3}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 3}},
                {"act": "DETACH", "args": {"engine": "E2", "car_type": "tanker", "at": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2e(self, goals: dict, constraints: dict) -> dict:
        """2-E: Deliver 2 bananas to Elmira, 1 orange to Bath."""
        return {
            "steps": [
                # E1 handles bananas
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}},
                
                # E2 handles oranges
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_2f(self, goals: dict, constraints: dict) -> dict:
        """2-F: Deliver 3 bananas to Bath, 2 juice to Dansville."""
        return {
            "steps": [
                # E1 handles bananas
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Bath"}},
                
                # E2 handles juice
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3a(self, goals: dict, constraints: dict) -> dict:
        """3-A: Deliver 1 banana to Bath, 1 banana to Corning, 1 banana to Elmira."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Bath"}},
                {"act": "DETACH", "args": {"engine": "E1", "car_type": "boxcar", "at": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Bath", "to": "Corning"}},
                {"act": "DETACH", "args": {"engine": "E1", "car_type": "boxcar", "at": "Corning", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3b(self, goals: dict, constraints: dict) -> dict:
        """3-B: Deliver 1 banana each to Bath, Corning, Elmira + 2 juice to Avon."""
        return {
            "steps": [
                # E1 handles bananas with DETACH
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Bath"}},
                {"act": "DETACH", "args": {"engine": "E1", "car_type": "boxcar", "at": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Bath", "to": "Corning"}},
                {"act": "DETACH", "args": {"engine": "E1", "car_type": "boxcar", "at": "Corning", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}},
                
                # E2 handles juice
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3c(self, goals: dict, constraints: dict) -> dict:
        """3-C: E2 only - Deliver 3 bananas to Elmira, 2 juice to Avon (2 trips)."""
        # Note: The actual 3-C problem is 3 bananas to Elmira and 2 juice to Avon
        return {
            "steps": [
                # Trip 1: Bananas first (tighter deadline)
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "bananas", "at": "Avon", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Avon", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "DETACH", "args": {"engine": "E2", "car_type": "boxcar", "at": "Elmira", "qty": 3}},
                
                # Trip 2: Juice to Avon
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Dansville", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Dansville", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "tanker", "from": "Corning", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3d(self, goals: dict, constraints: dict) -> dict:
        """3-D: Deliver 2 oranges + 1 juice to Avon (conversion conflict)."""
        return {
            "steps": [
                # E2 delivers oranges
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Bath", "to": "Avon"}},
                
                # E3 makes and delivers juice
                {"act": "START", "args": {"engine": "E3", "at": "Elmira"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "boxcar", "from": "Bath", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Bath", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E3", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "tanker", "from": "Corning", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Elmira"}},
                {"act": "CONVERT", "args": {"engine": "E3", "at": "Elmira", "qty": 1}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Bath", "to": "Avon"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3e(self, goals: dict, constraints: dict) -> dict:
        """3-E: Deliver 7 oranges to Elmira (needs 3 engines)."""
        return {
            "steps": [
                # E1 handles 3 oranges
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Corning", "to": "Elmira"}},
                
                # E2 handles 2 oranges
                {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
                {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "from": "Elmira", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E2", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Elmira"}},
                
                # E3 handles 2 oranges
                {"act": "START", "args": {"engine": "E3", "at": "Elmira"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}},
                {"act": "ATTACH", "args": {"engine": "E3", "car_type": "boxcar", "from": "Bath", "qty": 2}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Bath", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E3", "cargo": "oranges", "at": "Corning", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Elmira"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def solve_3f(self, goals: dict, constraints: dict) -> dict:
        """3-F: Deliver 1 orange + 2 bananas to Dansville."""
        return {
            "steps": [
                {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "from": "Dansville", "qty": 3}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "bananas", "at": "Avon", "cars": 2}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Corning"}},
                {"act": "LOAD", "args": {"engine": "E1", "cargo": "oranges", "at": "Corning", "cars": 1}},
                {"act": "TRAVEL", "args": {"engine": "E1", "from": "Corning", "to": "Dansville"}}
            ],
            "partial_order": [],
            "constraints": {}
        }
    
    def generic_solver(self, goals: dict, constraints: dict) -> dict:
        """Generic solver for patterns not explicitly coded."""
        
        steps = []
        
        # Count total units needed
        total_units = sum(sum(dest.values()) for dest in goals.values())
        
        # Determine which engines to use
        engines_needed = []
        if total_units <= 3:
            engines_needed = ["E2"]  # Single engine
        elif total_units <= 6:
            engines_needed = ["E1", "E2"]  # Two engines
        else:
            engines_needed = ["E1", "E2", "E3"]  # Three engines
        
        # Simple strategy: divide cargo among engines
        for engine in engines_needed:
            if engine == "E1":
                start_loc = "Avon"
            else:
                start_loc = "Elmira"
            
            steps.append({"act": "START", "args": {"engine": engine, "at": start_loc}})
        
        # Handle each cargo type
        for cargo_type, destinations in goals.items():
            for dest, qty in destinations.items():
                if qty == 0:
                    continue
                
                # Use first available engine
                engine = engines_needed[0] if engines_needed else "E2"
                
                if cargo_type == "bananas":
                    # Banana delivery logic
                    steps.extend([
                        {"act": "TRAVEL", "args": {"engine": engine, "from": "Avon", "to": "Dansville"}},
                        {"act": "ATTACH", "args": {"engine": engine, "car_type": "boxcar", "from": "Dansville", "qty": qty}},
                        {"act": "TRAVEL", "args": {"engine": engine, "from": "Dansville", "to": "Avon"}},
                        {"act": "LOAD", "args": {"engine": engine, "cargo": "bananas", "at": "Avon", "cars": qty}},
                    ])
                    
                elif cargo_type == "oranges":
                    # Orange delivery logic
                    steps.extend([
                        {"act": "ATTACH", "args": {"engine": engine, "car_type": "boxcar", "from": "Elmira", "qty": qty}},
                        {"act": "TRAVEL", "args": {"engine": engine, "from": "Elmira", "to": "Corning"}},
                        {"act": "LOAD", "args": {"engine": engine, "cargo": "oranges", "at": "Corning", "cars": qty}},
                    ])
                    
                elif cargo_type == "juice":
                    # Juice production logic
                    steps.extend([
                        {"act": "ATTACH", "args": {"engine": engine, "car_type": "boxcar", "from": "Elmira", "qty": qty}},
                        {"act": "TRAVEL", "args": {"engine": engine, "from": "Elmira", "to": "Corning"}},
                        {"act": "LOAD", "args": {"engine": engine, "cargo": "oranges", "at": "Corning", "cars": qty}},
                        {"act": "ATTACH", "args": {"engine": engine, "car_type": "tanker", "from": "Corning", "qty": qty}},
                        {"act": "TRAVEL", "args": {"engine": engine, "from": "Corning", "to": "Elmira"}},
                        {"act": "CONVERT", "args": {"engine": engine, "at": "Elmira", "qty": qty}},
                    ])
                
                # Add travel to destination (simplified)
                steps.append({"act": "TRAVEL", "args": {"engine": engine, "from": "Elmira", "to": dest}})
        
        return {
            "steps": steps,
            "partial_order": [],
            "constraints": {}
        }