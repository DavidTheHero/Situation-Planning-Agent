from __future__ import annotations
from typing import Dict, Tuple

# Travel times (hours) — legal edges only
EDGE_HOURS: Dict[Tuple[str,str], int] = {
    ("Avon","Dansville"): 3, ("Dansville","Avon"): 3,
    ("Avon","Bath"): 4, ("Bath","Avon"): 4,
    ("Dansville","Corning"): 1, ("Corning","Dansville"): 1,
    ("Corning","Bath"): 2, ("Bath","Corning"): 2,
    ("Corning","Elmira"): 2, ("Elmira","Corning"): 2,
}

# Facilities (where loading/processing is allowed)
FACILITY = {
    "Avon": {"banana_warehouse": True},
    "Corning": {"orange_warehouse": True},
    "Elmira": {"oj_factory": True},
    "Bath": {},
    "Dansville": {},
}

# Initial assets (global totals; problems may reposition via description)
INITIAL = {
    "engines": {"E1":"Avon","E2":"Elmira","E3":"Elmira"},
    # cars by location, with payload state: "empty" | "oranges" | "bananas" | "juice"
    "boxcars": {
        "Dansville": [{"payload":"empty"} for _ in range(3)],
        "Bath":      [{"payload":"empty"} for _ in range(2)],
        "Elmira":    [{"payload":"empty"} for _ in range(2)],
        "Avon": [], "Corning": []
    },
    "tankers": {
        "Corning": [{"payload":"empty"} for _ in range(3)],
        "Avon": [], "Bath": [], "Dansville": [], "Elmira": []
    }
}

# Operation durations (hours)
DUR = {
    "ATTACH": 0,
    "DETACH": 0,
    "LOAD": 1,
    "UNLOAD": 1,
    "CONVERT": 1,
    "WAIT": 0,      # WAIT uses explicit +Nh in the step
    "TRAVEL": None, # taken from EDGE_HOURS
}

# Engine capacity: max number of *loaded* cars (boxcars with oranges/bananas, or tankers with OJ)
MAX_LOADED_CAPACITY = 3
