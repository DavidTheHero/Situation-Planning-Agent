# minimal_test.py - Option 1: Enhanced signature approach (restored working version)
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os
import dspy
from dotenv import load_dotenv

from checker import check_plan

# Load environment variables
load_dotenv()

class PlanFromGoalSig(dspy.Signature):
    """Return ONLY valid JSON for the plan. No prose. Use the schema strictly.

    CAPACITY CONSTRAINT (NEVER VIOLATE):
    Maximum 3 LOADED cars per engine. LOADED = boxcars with cargo + tankers with juice.
    If total needed > 3 loaded cars, MUST use multiple engines.
    
    MEDIUM PROBLEM SUCCESS STRATEGY:
    
    ENGINE STATE TRACKING:
    - Engines MOVE during plan execution - track their location after each TRAVEL
    - Before ATTACH/LOAD: check if engine is at required location
    - If engine not at correct location: add TRAVEL step first
    - Example: E2@Elmira wants to ATTACH at Corning -> TRAVEL(E2:Elmira->Corning) first
    
    RESOURCE INVENTORY AWARENESS:
    Current yard inventory (CHECK BEFORE ATTACH):
    - Corning: 3 empty_tankers, 0 empty_boxcars (NEVER attach boxcars here!)
    - Elmira: 2 empty_boxcars, 0 empty_tankers (start here for boxcars!)
    - Bath: 2 empty_boxcars, 0 empty_tankers (OFTEN UNUSED - key for E3!)
    - Dansville: 3 empty_boxcars, 0 empty_tankers (additional boxcar source)
    - Avon: 0 empty_boxcars, 0 empty_tankers
    
    ATTACHMENT RULES (CRITICAL):
    - For boxcars: Prefer Elmira first (engines start there), then Bath/Dansville for additional
    - For tankers: ONLY attach at Corning
    - NEVER attach more cars than available at location
    - NEVER attach boxcars at Corning/Avon (they have 0 boxcars)
    - NEVER attach tankers at Elmira/Bath/Dansville/Avon (they have 0 tankers)
    - USE BATH BOXCARS: Bath has 2 boxcars often ignored by plans - use for E3!
    
    CARGO WAREHOUSE LOCATIONS (CRITICAL):
    - bananas: ONLY load at Avon warehouse
    - oranges: ONLY load at Corning warehouse  
    - juice/OJ: ONLY available at Corning OR must CONVERT from oranges
    
    MULTI-ENGINE COORDINATION (fixes goal shortfall):
    ENGINE ROLE ASSIGNMENT:
    - E1@Avon: Handle banana deliveries (closest to banana warehouse)
    - E2@Elmira: Handle juice conversions and orange deliveries  
    - E3@Elmira: Additional capacity for large deliveries (3+ units)
    
    CRITICAL: For 3+ juice units, MUST use multiple engines (capacity limit = 3 loaded per engine):
    - 1 juice: Use E2 only (1 orange boxcar + 1 tanker)
    - 2 juice: Use E2 only (2 orange boxcars + 2 tankers) 
    - 3 juice: Use E2 (2 juice) + E3 (1 juice) = 2 engines required
    - 4+ juice: Use E2 (2 juice) + E3 (2 juice) + additional engines
    
    CAPACITY RULE (CRITICAL - NEVER VIOLATE):
    - Maximum 3 LOADED cars per engine at any time
    - LOADED cars = boxcars with cargo (oranges/bananas) + tankers with juice
    - EMPTY cars do NOT count toward the 3-car limit
    - If total needed > 3 loaded cars, MUST split across multiple engines
    - Calculate: Count juice tankers + cargo boxcars for each engine
    - Use multiple engines if any single engine would exceed 3 loaded cars
    
    MULTI-ENGINE CAPACITY STRATEGY:
    - For 4+ total units: Split optimally across E2 and E3
    - E2 typically handles juice production + some cargo
    - E3 handles additional cargo to stay under capacity
    - Each engine max 3 loaded cars - plan accordingly
    
    NEVER exceed 3 loaded cars on any engine - plan will fail capacity check!
    
    GOAL VERIFICATION (CRITICAL):
    - Count total units needed vs planned deliveries
    - Ensure each engine's contribution adds up to exact goal requirements
    - Double-check quantities in ATTACH actions match delivery needs
    - Verify: Sum of all engine deliveries = Total goal requirements
    
    JUICE PRODUCTION SPLIT STRATEGY (for 3 juice units):
    E2 workflow: ATTACH(2 boxcars from Elmira) + TRAVEL(Corning) + ATTACH(2 tankers) + LOAD(oranges,2) + TRAVEL(Elmira) + CONVERT(qty=2) + TRAVEL(Bath)
    E3 workflow: TRAVEL(Bath) + ATTACH(1 boxcar from Bath) + TRAVEL(Corning) + ATTACH(1 tanker) + LOAD(oranges,1) + TRAVEL(Elmira) + CONVERT(qty=1) + TRAVEL(Bath)
    
    LARGE DELIVERY STRATEGY (>3 units):
    - Split across multiple engines due to 3-car capacity limit
    - Each engine: ATTACH all needed boxcars at starting location FIRST
    - For 4 oranges: E2 takes 2 cars from Elmira, E3 takes 2 cars from Bath
    - NEVER split ATTACH operations - get all cars at start location
    - BATH STRATEGY: E3 travels Elmira→Corning→Bath to get Bath boxcars when Elmira depleted
    
    PARALLEL EXECUTION TEMPLATE:
    E1 Workflow: START(E1@Avon) -> TRAVEL(get resources) -> ATTACH(boxcars) -> 
                 TRAVEL(Avon) -> LOAD(bananas) -> TRAVEL(destination)
    E2 Workflow: START(E2@Elmira) -> ATTACH(2 boxcars@Elmira) -> TRAVEL(Corning) -> 
                 LOAD(oranges,2) -> TRAVEL(destination)
    E3 Workflow: START(E3@Elmira) -> TRAVEL(Bath) -> ATTACH(2 boxcars@Bath) -> 
                 TRAVEL(Corning) -> LOAD(oranges,2) -> TRAVEL(destination)
    
    DEADLINE OPTIMIZATION (fixes 58% of failures):
    - Use shortest valid paths (check network topology)
    - Minimize detours before loading cargo
    - Start engines in parallel when possible
    - Consider total travel time: Avon<->Dansville=3h, Corning<->Bath=4h
    
    ROUTING RULES (CRITICAL - fixes illegal edge errors):
    Adjacent connections ONLY (never skip intermediate stops):
    - Avon ↔ Dansville
    - Dansville ↔ Corning  
    - Corning ↔ Bath
    - Bath ↔ Avon
    - Elmira ↔ Corning
    - NO DIRECT Elmira ↔ Dansville connection!
    - NO DIRECT Elmira ↔ Bath connection! (must go via Corning)
    - NO DIRECT Elmira ↔ Avon connection! (must go via Corning→Bath)
    
    MULTI-HOP ROUTING (must break into individual TRAVEL steps):
    - Avon to Corning: TRAVEL(Avon→Dansville) + TRAVEL(Dansville→Corning)
    - Elmira to Avon: TRAVEL(Elmira→Corning) + TRAVEL(Corning→Bath) + TRAVEL(Bath→Avon)
    - Elmira to Bath: TRAVEL(Elmira→Corning) + TRAVEL(Corning→Bath) [CRITICAL FOR E3!]
    - Bath to Dansville: TRAVEL(Bath→Corning) + TRAVEL(Corning→Dansville)
    
    NEVER use direct travel between non-adjacent locations. Always break multi-hop routes into individual TRAVEL steps.
    CRITICAL: Elmira to Bath requires 2 steps: Elmira→Corning→Bath
    
    JUICE PRODUCTION WORKFLOW (fixes CONVERT failures):
    For juice delivery: oranges -> CONVERT -> juice workflow
    CRITICAL SEQUENCE (must follow exactly):
    1. START at Elmira (E2/E3 location)
    2. ATTACH empty boxcar at Elmira (ONLY location with boxcars)
    3. TRAVEL to Corning  
    4. LOAD oranges into boxcar at Corning
    5. ATTACH empty tanker at Corning (ONLY location with tankers)
    6. TRAVEL back to Elmira
    7. CONVERT (transforms oranges->juice, empties tanker->juice tanker)
    8. TRAVEL to final destination
    
    JUICE RESOURCE RULE: 
    - Boxcars ONLY available at Elmira/Bath - attach there FIRST
    - Tankers ONLY available at Corning - attach there AFTER loading oranges
    - Never try to attach boxcars at Corning (has 0 boxcars)
    
    CRITICAL ACTION FORMAT RULES:
    - START: {"act": "START", "args": {"engine": "E2", "at": "Elmira"}}
    - ATTACH: {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "qty": 2, "from": "Elmira"}}
    - DETACH: {"act": "DETACH", "args": {"engine": "E2", "car_type": "boxcar", "qty": 1, "at": "Bath"}}
    - LOAD: {"act": "LOAD", "args": {"engine": "E2", "at": "Corning", "cargo": "oranges", "cars": 2}}
    - TRAVEL: {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}}
    - CONVERT: {"act": "CONVERT", "args": {"engine": "E2", "at": "Elmira"}}
    
    DETACH USAGE (for multi-destination deliveries):
    - Use DETACH to drop off specific cars at intermediate stops
    - DETACH allows partial delivery of loaded cargo
    - Example: If carrying 3 loaded boxcars, can DETACH 1 at each stop
    - Critical for problems requiring delivery to multiple destinations
    
    NEVER USE: "cars": ["boxcar"], "cargo": "OJ", "at" for ATTACH
    ALWAYS USE: "car_type": "boxcar", "qty": 1, "cargo": "oranges", "from" for ATTACH
    
    PRE-LOADED CARS (when problem mentions cars "waiting" at location):
    - Use payload parameter for pre-loaded cars: "payload": "oranges"
    - Example: {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "qty": 1, "from": "Corning", "payload": "oranges"}}
    - These boxcars are already loaded with cargo, no LOAD action needed
    
    FOR 3+ JUICE UNITS - CRITICAL MULTI-ENGINE PATTERN:
    E2: ATTACH(2 boxcars@Elmira) + TRAVEL(Corning) + LOAD(oranges,2) + ATTACH(2 tankers) + TRAVEL(Elmira) + CONVERT(qty=2) + TRAVEL(Bath)
    E3: TRAVEL(Bath) + ATTACH(1 boxcar@Bath) + TRAVEL(Corning) + LOAD(oranges,1) + ATTACH(1 tanker) + TRAVEL(Elmira) + CONVERT(qty=1) + TRAVEL(Bath)
    
    COMMON FAILURE PREVENTION:
    - Never ATTACH more cars than yard has available
    - Never ATTACH boxcars at Corning (has 0 boxcars - only tankers)
    - Never ATTACH tankers at Elmira/Bath/Dansville (have 0 tankers)
    - Never LOAD cargo at wrong warehouse
    - Never CONVERT without both loaded oranges AND empty tankers attached
    - Never exceed 3 loaded units per engine capacity
    - Always add TRAVEL between different locations
    - ATTACH all needed cars at Dansville/Bath before traveling to load location
    
    STEP-BY-STEP PLANNING CHECKLIST:
    1. Identify cargo types needed and destinations
    2. Assign engines based on proximity to resources
    3. Plan parallel workflows to minimize total time
    4. For each step: verify engine location and resource availability
    5. Add TRAVEL steps when engine needs to move
    6. Check final plan meets deadlines and goals
    """

    goal        = dspy.InputField(desc="Problem description plus rules/context, already wrapped.")
    constraints = dspy.InputField(desc="JSON-encoded constraints parsed from the problem text.")
    plan_json   = dspy.OutputField(desc="""Return ONLY a JSON object with exact schema:
{
  "steps": [
    {"act": "START", "args": {"engine": "E1", "at": "Avon"}},
    {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Dansville"}},
    {"act": "ATTACH", "args": {"engine": "E1", "car_type": "boxcar", "qty": 2, "from": "Dansville"}},
    {"act": "TRAVEL", "args": {"engine": "E1", "from": "Dansville", "to": "Avon"}},
    {"act": "LOAD", "args": {"engine": "E1", "at": "Avon", "cargo": "bananas", "cars": 2}},
    {"act": "TRAVEL", "args": {"engine": "E1", "from": "Avon", "to": "Corning"}}
  ],
  "partial_order": [],
  "constraints": {}
}

CRITICAL: Use "act" not "action". Use "args" not separate fields. Return direct JSON, not nested under "plan_json" key.
LOAD format: {"act": "LOAD", "args": {"engine": "E1", "at": "location", "cargo": "bananas", "cars": 2}}
Include "cars" parameter in LOAD to specify quantity loaded.

ATTACH format: {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "qty": 2, "from": "Elmira"}}
For pre-loaded cars add: "payload": "oranges"

EXAMPLE FORMAT (NOT a specific solution - adapt to your problem):
For problems requiring multiple engines due to capacity:

{
  "steps": [
    {"act": "START", "args": {"engine": "E2", "at": "Elmira"}},
    {"act": "ATTACH", "args": {"engine": "E2", "car_type": "boxcar", "qty": 1, "from": "Elmira"}},
    {"act": "TRAVEL", "args": {"engine": "E2", "from": "Elmira", "to": "Corning"}},
    {"act": "LOAD", "args": {"engine": "E2", "at": "Corning", "cargo": "oranges", "cars": 1}},
    {"act": "TRAVEL", "args": {"engine": "E2", "from": "Corning", "to": "Bath"}},
    {"act": "START", "args": {"engine": "E3", "at": "Elmira"}},
    {"act": "TRAVEL", "args": {"engine": "E3", "from": "Elmira", "to": "Corning"}},
    {"act": "ATTACH", "args": {"engine": "E3", "car_type": "boxcar", "qty": 2, "from": "Bath"}},
    {"act": "TRAVEL", "args": {"engine": "E3", "from": "Bath", "to": "Corning"}},
    {"act": "LOAD", "args": {"engine": "E3", "at": "Corning", "cargo": "oranges", "cars": 2}},
    {"act": "TRAVEL", "args": {"engine": "E3", "from": "Corning", "to": "Bath"}}
  ]
}""")


def call_dspy_planner(goal_text: str, constraints: dict) -> dict:
    """Call DSPy planner using the PlanFromGoalSig signature."""
    
    try:
        # Create the planner using the signature
        planner = dspy.Predict(PlanFromGoalSig)
        
        print(f"Debug: Calling DSPy Option 1 with goal: {goal_text}")
        print(f"Debug: Constraints: {constraints}")
        
        # Call the planner with goal and constraints
        result = planner(
            goal=goal_text,
            constraints=json.dumps(constraints)
        )
        
        print(f"Debug: DSPy result type = {type(result)}")
        print(f"Debug: DSPy result = {result}")
        
        # Extract plan from result
        if hasattr(result, 'plan_json'):
            plan_data = result.plan_json
            print(f"Debug: plan_json = {plan_data}")
            
            # If it's a string, try to parse as JSON
            if isinstance(plan_data, str):
                try:
                    plan_data = json.loads(plan_data)
                except json.JSONDecodeError:
                    # Try to extract JSON from string
                    import re
                    json_match = re.search(r'\{.*\}', plan_data, re.DOTALL)
                    if json_match:
                        plan_data = json.loads(json_match.group())
                    else:
                        print("Debug: Could not parse JSON from string")
                        return {"steps": [], "partial_order": [], "constraints": {}}
            
            # Handle different plan structures
            if isinstance(plan_data, dict):
                # Direct steps format
                if 'steps' in plan_data:
                    return {
                        "steps": plan_data['steps'],
                        "partial_order": plan_data.get('partial_order', []),
                        "constraints": plan_data.get('constraints', {})
                    }
                # Nested plan_json format
                elif 'plan_json' in plan_data and isinstance(plan_data['plan_json'], dict):
                    inner_plan = plan_data['plan_json']
                    if 'steps' in inner_plan:
                        return {
                            "steps": inner_plan['steps'],
                            "partial_order": inner_plan.get('partial_order', []),
                            "constraints": inner_plan.get('constraints', {})
                        }
                        
        print(f"Debug: Could not extract valid plan structure")
        
    except Exception as e:
        print(f"DSPy planning error: {e}")
        import traceback
        traceback.print_exc()
    
    # Return empty plan if anything fails
    return {"steps": [], "partial_order": [], "constraints": {}}


# Copy utility functions
def print_section(title):
    print(f"========== {title} ==========")

def pretty(obj):
    result = json.dumps(obj, indent=2, ensure_ascii=False)
    # Clean up common Unicode escapes for better readability
    result = result.replace('\\u2192', '→')
    result = result.replace('\\u22643', '≤3')
    return result

def pretty_constraints_with_hours(constraints):
    """Convert constraints to show hours instead of raw numbers."""
    if not isinstance(constraints, dict):
        return pretty(constraints)
    
    result = {}
    for key, value in constraints.items():
        if key == "deliver_deadlines" and isinstance(value, dict):
            result[key] = {}
            for cargo, destinations in value.items():
                result[key][cargo] = {}
                for dest, hours in destinations.items():
                    result[key][cargo][dest] = f"{hours} hours"
        else:
            result[key] = value
    
    return pretty(result)

def summarize_steps(plan):
    """Create a numbered summary of plan steps."""
    if not isinstance(plan, dict) or 'steps' not in plan:
        return "(no steps)"
    
    lines = []
    for i, step in enumerate(plan['steps'], 1):
        if isinstance(step, dict) and 'act' in step:
            act = step['act']
            args = {k: v for k, v in step.items() if k != 'act'}
            lines.append(f"{i:02d}. {act} {args}")
        else:
            lines.append(f"{i:02d}. UNKNOWN {step}")
    
    # Truncate if too many steps
    if len(lines) > 10:
        lines = lines[:10] + [f"... ({len(plan['steps']) - 10} more steps)"]
    
    return "\n".join(lines)

def write_jsonl(path, data):
    """Append a JSON line to a file."""
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def run_problem(pid: str, goal_text: str, plan_data: dict, log_path: str) -> dict:
    """Run planning with natural description for DSPy and pre-parsed data for checking."""
    
    print_section(f"PROBLEM: {pid}")
    print(f"DESCRIPTION: {goal_text}")
    print_section("")
    
    # Get pre-parsed goals and constraints
    goals = plan_data.get("goal", {})
    constraints = plan_data.get("constraints", {})
    
    print(f"GOALS FROM all_plans_description.json: {goals}")
    print(f"CONSTRAINTS FROM all_plans_description.json: {constraints}")
    print_section("")
    print(f"Feeding to DSPy Option 1: {goal_text}")
    
    # Generate plan using DSPy signature
    plan = call_dspy_planner(goal_text, constraints)
    
    # Check plan using pre-parsed goals and constraints
    # Extract overrides from plan_data if present
    overrides = plan_data.get("overrides", {})
    
    try:
        res = check_plan(
            plan=plan,
            problem_desc=goal_text,
            goals=goals,
            constraints=constraints,
            overrides=overrides
        )
    except Exception as e:
        print(f"Error checking plan: {e}")
        res = {"goal_achieved": 0, "violations": [f"Checker error: {e}"]}
    
    # Display results
    print_section(f"PROBLEM {pid}")
    print("Goal (raw description):")
    print(goal_text)
    print_section("Goals (from all_plans_description.json)")
    print(pretty(goals))
    print_section("Constraints (from all_plans_description.json)")
    print(pretty_constraints_with_hours(constraints))
    print_section("Generated Plan")
    print(summarize_steps(plan))
    print_section("Checker Result")
    print(pretty(res))
    
    # Log structured data
    write_jsonl(log_path, {
        "problem_id": pid,
        "goal_text": goal_text,
        "goals": goals,
        "constraints": constraints,
        "plan": plan,
        "result": res,
        "goal_achieved": res.get("goal_achieved", 0),
        "violations": res.get("violations", [])
    })
    
    return {"problem_id": pid, "plan": plan, "result": res}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated problem IDs to test")
    parser.add_argument("--log", default="minimal_test_option1.jsonl", help="Output log file")
    args = parser.parse_args()
    
    # Set up DSPy with your LLM
    lm = dspy.LM('openai/gpt-4o', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)
    
    # Load problem descriptions and parsed data  
    with open("all_plans_description.json", "r") as f:
        plans_data = json.load(f)
    
    # Load problem descriptions from CISD
    problems_data = {}
    if os.path.exists("cisd_cleaned.json"):
        with open("cisd_cleaned.json", "r") as f:
            cisd_data = json.load(f)
            problems_data = {pid: data.get("description", f"Problem {pid}") for pid, data in cisd_data.items()}
    else:
        print("Warning: cisd_cleaned.json not found. Using placeholder descriptions.")
    
    # Process specified problems
    if args.ids:
        problem_ids = [pid.strip() for pid in args.ids.split(",")]
    else:
        problem_ids = list(problems_data.keys())
    
    results = []
    skipped = 0
    
    for pid in problem_ids:
        if pid not in problems_data:
            print(f"Warning: Problem {pid} not found in descriptions")
            skipped += 1
            continue
            
        if pid not in plans_data:
            print(f"Warning: Problem {pid} not found in parsed plans")  
            skipped += 1
            continue
            
        desc = problems_data[pid]
        
        try:
            result = run_problem(
                pid=pid,
                goal_text=desc,
                plan_data=plans_data[pid],
                log_path=args.log
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing problem {pid}: {e}")
            skipped += 1
    
    print(f"\nCompleted: {len(results)} problems processed, {skipped} skipped.")

if __name__ == "__main__":
    main()
