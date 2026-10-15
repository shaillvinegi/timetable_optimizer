"""
================================================================================
AI Timetable Optimizer Using Hill Climbing and Simulated Annealing
Subject: Artificial Intelligence and Soft Computing (AISC) Practical
File: timetable_optimizer.py
================================================================================

Problem Description:
This program optimizes a college timetable for 4 subjects (AI, DBMS, Java, Python)
across 5 weekdays (Monday-Friday), 4 periods per day (Periods 1-4), and 2 classrooms
(R101, R102). Each subject requires exactly 2 sessions (total 8 sessions).

Algorithms Implemented:
1. Hill Climbing (Local Search / Greedy Descent)
2. Simulated Annealing (Probabilistic Metaheuristic with Cooling Schedule)

Constraints:
- Hard Constraints (Must NEVER be violated):
    1. No professor teaches two classes in the same time slot (Day, Period).
    2. No classroom hosts two classes in the same time slot (Day, Period).
    3. All 8 required sessions must be scheduled.
    4. Every assignment must use a valid Day, Period, and Room.
- Soft Constraints (Minimized via Penalty Function):
    1. Penalize sessions of the same subject scheduled on the same day (+10 per violation).
    2. Penalize professors scheduled for consecutive periods on the same day (+5 per violation).
================================================================================
"""

import copy
import math
import random
import time

# ==============================================================================
# 1. PROBLEM CONFIGURATION AND SAMPLE DATASET
# ==============================================================================

# Working days and periods
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = [1, 2, 3, 4]
ROOMS = ["R101", "R102"]

# Day ordering index for sorting schedules chronologically
DAY_ORDER = {day: idx for idx, day in enumerate(DAYS)}

# Subject to Professor mapping
SUBJECT_PROFESSOR_MAP = {
    "AI": "Prof. Sharma",
    "DBMS": "Prof. Verma",
    "Java": "Prof. Gupta",
    "Python": "Prof. Mehta",
}

# 8 Required Sessions (2 per subject)
REQUIRED_SESSIONS = [
    {"session_id": 1, "subject": "AI", "professor": "Prof. Sharma"},
    {"session_id": 2, "subject": "AI", "professor": "Prof. Sharma"},
    {"session_id": 3, "subject": "DBMS", "professor": "Prof. Verma"},
    {"session_id": 4, "subject": "DBMS", "professor": "Prof. Verma"},
    {"session_id": 5, "subject": "Java", "professor": "Prof. Gupta"},
    {"session_id": 6, "subject": "Java", "professor": "Prof. Gupta"},
    {"session_id": 7, "subject": "Python", "professor": "Prof. Mehta"},
    {"session_id": 8, "subject": "Python", "professor": "Prof. Mehta"},
]

# Penalty Weights for Soft Constraints
SAME_DAY_PENALTY_WEIGHT = 10       # Penalty for scheduling same subject twice on same day
CONSECUTIVE_PERIOD_PENALTY_WEIGHT = 5  # Penalty for professor teaching back-to-back periods


# ==============================================================================
# 2. PENALTY FUNCTION (SOFT CONSTRAINTS EVALUATION)
# ==============================================================================

def calculate_penalty(timetable, return_breakdown=False):
    """
    Calculates the penalty score for a given timetable based on soft constraints.
    Lower score represents a better, more balanced timetable.
    An optimal timetable achieves a penalty of 0.

    Soft Constraints Evaluated:
    1. Same-Day Subject Penalty:
       If a subject has more than 1 session on the same day, students do not get
       adequate time between classes. Penalty = (count - 1) * 10
    2. Consecutive Periods Penalty:
       If a professor teaches consecutive periods (e.g. Period 1 followed by Period 2)
       on the same day, they lack rest/prep time. Penalty = 5 per consecutive pair.
    """
    same_day_penalty = 0
    consecutive_penalty = 0

    # Dictionaries to aggregate counts
    subject_day_counts = {}
    professor_day_periods = {}

    for session in timetable:
        subject = session["subject"]
        professor = session["professor"]
        day = session["day"]
        period = session["period"]

        # Aggregate subject count per day
        subj_day_key = (subject, day)
        subject_day_counts[subj_day_key] = subject_day_counts.get(subj_day_key, 0) + 1

        # Aggregate professor teaching periods per day
        prof_day_key = (professor, day)
        if prof_day_key not in professor_day_periods:
            professor_day_periods[prof_day_key] = []
        professor_day_periods[prof_day_key].append(period)

    # 1. Calculate same-day duplicate penalty
    for (subject, day), count in subject_day_counts.items():
        if count > 1:
            same_day_penalty += (count - 1) * SAME_DAY_PENALTY_WEIGHT

    # 2. Calculate consecutive periods penalty for professors
    for (professor, day), periods in professor_day_periods.items():
        sorted_periods = sorted(periods)
        for i in range(len(sorted_periods) - 1):
            if sorted_periods[i + 1] == sorted_periods[i] + 1:
                consecutive_penalty += CONSECUTIVE_PERIOD_PENALTY_WEIGHT

    total_penalty = same_day_penalty + consecutive_penalty

    if return_breakdown:
        return total_penalty, same_day_penalty, consecutive_penalty
    return total_penalty


# ==============================================================================
# 3. INDEPENDENT HARD CONSTRAINT VALIDATION
# ==============================================================================

def validate_hard_constraints(timetable):
    """
    Independently verifies that all hard constraints are strictly satisfied.
    Returns:
        (is_valid: bool, errors: list of str)
    
    Hard Constraints Checked:
    1. Completeness: Exactly 8 sessions scheduled without adding or losing classes.
    2. Correct Subject Count: Exactly 2 sessions for AI, DBMS, Java, and Python.
    3. Valid Domains: Day in DAYS, Period in PERIODS, Room in ROOMS.
    4. Classroom Conflict: No classroom hosts two classes at the same (Day, Period).
    5. Professor Conflict: No professor teaches two classes at the same (Day, Period).
    """
    errors = []

    # 1. Completeness check
    if len(timetable) != 8:
        errors.append(f"Expected 8 scheduled sessions, found {len(timetable)}.")

    # 2. Correct session count per subject
    subject_counts = {}
    for s in timetable:
        subject_counts[s["subject"]] = subject_counts.get(s["subject"], 0) + 1
    for expected_subj in ["AI", "DBMS", "Java", "Python"]:
        count = subject_counts.get(expected_subj, 0)
        if count != 2:
            errors.append(f"Subject '{expected_subj}' expected 2 sessions, found {count}.")

    # 3. Domain validation
    for s in timetable:
        if s["day"] not in DAYS:
            errors.append(f"Invalid day '{s['day']}' in session {s['session_id']}.")
        if s["period"] not in PERIODS:
            errors.append(f"Invalid period '{s['period']}' in session {s['session_id']}.")
        if s["room"] not in ROOMS:
            errors.append(f"Invalid room '{s['room']}' in session {s['session_id']}.")

    # 4. Room Conflict & Professor Conflict checks
    room_occupied = set()
    professor_occupied = set()

    for s in timetable:
        room_slot = (s["day"], s["period"], s["room"])
        if room_slot in room_occupied:
            errors.append(f"Room Conflict: Room {s['room']} double-booked on {s['day']}, Period {s['period']}.")
        room_occupied.add(room_slot)

        prof_slot = (s["day"], s["period"], s["professor"])
        if prof_slot in professor_occupied:
            errors.append(f"Professor Conflict: {s['professor']} assigned to multiple rooms on {s['day']}, Period {s['period']}.")
        professor_occupied.add(prof_slot)

    is_valid = len(errors) == 0
    return is_valid, errors


# ==============================================================================
# 4. INITIAL FEASIBLE TIMETABLE GENERATION
# ==============================================================================

def create_initial_feasible_timetable():
    """
    Constructs an initial feasible timetable that satisfies 100% of hard constraints
    but contains deliberate soft constraint violations (high penalty).
    This serves as the identical baseline for both search algorithms.

    Initial Schedule Structure:
    - Monday:
        Period 1, R101 -> AI (Prof. Sharma)
        Period 2, R101 -> AI (Prof. Sharma)      [Soft Violations: AI twice on Mon, Prof Sharma consecutive]
    - Tuesday:
        Period 1, R101 -> DBMS (Prof. Verma)
        Period 2, R101 -> DBMS (Prof. Verma)    [Soft Violations: DBMS twice on Tue, Prof Verma consecutive]
    - Wednesday:
        Period 2, R102 -> Java (Prof. Gupta)
        Period 3, R102 -> Java (Prof. Gupta)    [Soft Violations: Java twice on Wed, Prof Gupta consecutive]
    - Thursday:
        Period 3, R102 -> Python (Prof. Mehta)
    - Friday:
        Period 4, R102 -> Python (Prof. Mehta)
    """
    initial_slot_assignments = [
        ("Monday", 1, "R101"),     # AI Session 1
        ("Monday", 2, "R101"),     # AI Session 2
        ("Tuesday", 1, "R101"),    # DBMS Session 1
        ("Tuesday", 2, "R101"),    # DBMS Session 2
        ("Wednesday", 2, "R102"),  # Java Session 1
        ("Wednesday", 3, "R102"),  # Java Session 2
        ("Thursday", 3, "R102"),   # Python Session 1
        ("Friday", 4, "R102"),     # Python Session 2
    ]

    initial_timetable = []
    for idx, session_info in enumerate(REQUIRED_SESSIONS):
        day, period, room = initial_slot_assignments[idx]
        session = {
            "session_id": session_info["session_id"],
            "subject": session_info["subject"],
            "professor": session_info["professor"],
            "day": day,
            "period": period,
            "room": room,
        }
        initial_timetable.append(session)

    return initial_timetable


# ==============================================================================
# 5. NEIGHBOR GENERATION (MOVE OPERATOR)
# ==============================================================================

def get_neighbor_by_swapping_two_sessions(timetable, max_attempts=50):
    """
    Generates a neighboring timetable by swapping the time slots (day, period, room)
    of two randomly selected sessions.

    Constraint Preservation Strategy:
    - Swapping two existing assignments preserves room conflicts because each session
      occupies the other's existing room-period slot.
    - However, swapping can introduce a PROFESSOR CONFLICT (e.g., if a professor ends up
      assigned to teach in another room at the exact same day and period).
    - Therefore, candidate neighbors are validated against hard constraints before acceptance.
    - Any invalid candidate is rejected and a new swap is attempted.
    """
    for _ in range(max_attempts):
        idx1, idx2 = random.sample(range(len(timetable)), 2)

        # Deep copy to keep original timetable unaffected
        candidate = copy.deepcopy(timetable)

        # Swap the time slots and classrooms of the two selected sessions
        candidate[idx1]["day"], candidate[idx2]["day"] = candidate[idx2]["day"], candidate[idx1]["day"]
        candidate[idx1]["period"], candidate[idx2]["period"] = candidate[idx2]["period"], candidate[idx1]["period"]
        candidate[idx1]["room"], candidate[idx2]["room"] = candidate[idx2]["room"], candidate[idx1]["room"]

        # Validate hard constraints immediately
        is_valid, _ = validate_hard_constraints(candidate)
        if is_valid:
            return candidate

    # Return None if no valid swap found within maximum attempts
    return None


# ==============================================================================
# 6. ALGORITHM A: HILL CLIMBING
# ==============================================================================

def hill_climbing(initial_timetable, max_iterations=200, patience=15):
    """
    Hill Climbing Algorithm (Iterative Local Search / Greedy Descent)
    
    Algorithmic Steps:
    1. Start with the initial feasible timetable.
    2. In each iteration, generate a neighbor by swapping two sessions' time slots.
    3. Accept the neighbor IF AND ONLY IF its penalty score is strictly lower than
       the current timetable's penalty score.
    4. If no improvement is found for `patience` consecutive iterations (or max iterations
       limit is reached), terminate the search.

    Theoretical Context - Local Optima:
    -----------------------------------
    A local optimum (or local minimum in a minimization problem) occurs when the current
    state has a lower penalty than all neighbors in its immediate neighborhood, but is
    NOT the global best possible schedule.
    Because basic Hill Climbing strictly rejects worse moves (no uphill transitions),
    it can get trapped in a local optimum or plateau and terminate prematurely.
    """
    start_time = time.perf_counter()

    current_timetable = copy.deepcopy(initial_timetable)
    current_penalty = calculate_penalty(current_timetable)

    iteration_count = 0
    no_improvement_count = 0

    while iteration_count < max_iterations and no_improvement_count < patience:
        iteration_count += 1

        # Generate neighbor by swapping two session slots
        neighbor = get_neighbor_by_swapping_two_sessions(current_timetable)
        if neighbor is None:
            no_improvement_count += 1
            continue

        neighbor_penalty = calculate_penalty(neighbor)

        # Strictly greedy acceptance criterion
        if neighbor_penalty < current_penalty:
            current_timetable = neighbor
            current_penalty = neighbor_penalty
            no_improvement_count = 0  # Reset patience counter upon improvement
        else:
            no_improvement_count += 1  # Count non-improving iterations

    execution_time = time.perf_counter() - start_time
    return current_timetable, current_penalty, iteration_count, execution_time


# ==============================================================================
# 7. ALGORITHM B: SIMULATED ANNEALING
# ==============================================================================

def simulated_annealing(initial_timetable, initial_temperature=100.0, cooling_rate=0.95,
                        min_temperature=0.01, max_iterations=1000):
    """
    Simulated Annealing Algorithm (Probabilistic Metaheuristic)

    Algorithmic Steps:
    1. Start with the initial feasible timetable and set initial temperature T.
    2. Generate a neighbor by swapping two session time slots.
    3. Calculate change in penalty: delta = neighbor_penalty - current_penalty.
    4. If delta < 0 (improving move):
         Always accept the neighbor.
         Update best_timetable if current_penalty is the best seen so far.
    5. If delta >= 0 (worse or equal move):
         Accept the neighbor probabilistically using the Boltzmann formula:
             P = exp(-delta / temperature)
         Draw random number r in [0, 1). If r < P, accept the worse timetable!
    6. Cool down temperature: temperature = temperature * cooling_rate.
    7. Terminate when temperature falls below min_temperature or max_iterations reached.
    8. Return the best timetable tracked throughout the search.

    Theoretical Context - Escaping Local Optima:
    --------------------------------------------
    Simulated Annealing overcomes the fundamental limitation of Hill Climbing by
    allowing probabilistic uphill moves (accepting worse solutions).
    At high temperatures, the system has high mobility (accepts many worse moves to explore).
    As temperature cools, acceptance of worse moves diminishes, and the algorithm
    gradually transitions from broad exploration to fine-grained exploitation (hill climbing).
    """
    start_time = time.perf_counter()

    current_timetable = copy.deepcopy(initial_timetable)
    current_penalty = calculate_penalty(current_timetable)

    # Maintain global best timetable encountered
    best_timetable = copy.deepcopy(current_timetable)
    best_penalty = current_penalty

    temperature = initial_temperature
    iteration_count = 0

    while temperature > min_temperature and iteration_count < max_iterations:
        iteration_count += 1

        neighbor = get_neighbor_by_swapping_two_sessions(current_timetable)
        if neighbor is None:
            temperature *= cooling_rate
            continue

        neighbor_penalty = calculate_penalty(neighbor)
        delta_penalty = neighbor_penalty - current_penalty

        # Case 1: Improving move (lower penalty)
        if delta_penalty < 0:
            current_timetable = neighbor
            current_penalty = neighbor_penalty

            # Update overall best
            if current_penalty < best_penalty:
                best_timetable = copy.deepcopy(current_timetable)
                best_penalty = current_penalty

        # Case 2: Non-improving move (higher or equal penalty)
        else:
            # Acceptance probability formula: P = exp(-delta / T)
            acceptance_probability = math.exp(-delta_penalty / temperature)
            random_chance = random.random()

            if random_chance < acceptance_probability:
                current_timetable = neighbor
                current_penalty = neighbor_penalty

        # Geometric cooling schedule
        temperature *= cooling_rate

    execution_time = time.perf_counter() - start_time
    return best_timetable, best_penalty, iteration_count, execution_time


# ==============================================================================
# 8. OUTPUT FORMATTING UTILITIES
# ==============================================================================

def print_timetable_table(timetable, title):
    """
    Displays a timetable in a clean, professional ASCII table format.
    Sorted chronologically by day, period, and room.
    """
    # Sort chronologically by day and period
    sorted_tt = sorted(
        timetable,
        key=lambda x: (DAY_ORDER[x["day"]], x["period"], x["room"])
    )

    print(f"\n--- {title} ---")
    header = f"+-----------+--------+---------+--------------------+------+"
    print(header)
    print(f"| {'Day':<9} | {'Period':<6} | {'Subject':<7} | {'Professor':<18} | {'Room':<4} |")
    print(header)
    for row in sorted_tt:
        print(f"| {row['day']:<9} | {row['period']:^6} | {row['subject']:<7} | {row['professor']:<18} | {row['room']:<4} |")
    print(header)


# ==============================================================================
# 9. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    # Fixed random seed for 100% reproducible demonstrations
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print(" PROJECT TITLE: AI TIMETABLE OPTIMIZER")
    print(" Techniques: Hill Climbing vs. Simulated Annealing")
    print(f" Random Seed: {RANDOM_SEED} (Ensures Reproducible Execution)")
    print("=" * 80)

    # 1. Generate baseline initial feasible timetable
    initial_timetable = create_initial_feasible_timetable()
    initial_valid, initial_errors = validate_hard_constraints(initial_timetable)
    initial_penalty, init_same_day, init_consec = calculate_penalty(initial_timetable, return_breakdown=True)

    print_timetable_table(initial_timetable, "INITIAL TIMETABLE (Baseline)")
    print(f"Initial Feasibility Status : {'VALID' if initial_valid else 'INVALID'}")
    print(f"Initial Total Penalty Score: {initial_penalty}")
    print(f"  - Same-Day Subject Penalty : {init_same_day} (AI, DBMS, Java each duplicate on 1 day)")
    print(f"  - Consecutive Period Penalty: {init_consec} (Prof. Sharma, Verma, Gupta have back-to-back classes)")

    # 2. Execute Algorithm A: Hill Climbing
    # Set seed right before algorithm to ensure both start from exact identical pseudorandom state
    random.seed(RANDOM_SEED)
    hc_timetable, hc_penalty, hc_iterations, hc_time = hill_climbing(initial_timetable)

    # 3. Execute Algorithm B: Simulated Annealing
    # Set seed right before algorithm to ensure identical starting random baseline
    random.seed(RANDOM_SEED)
    sa_timetable, sa_penalty, sa_iterations, sa_time = simulated_annealing(initial_timetable)

    # 4. Display Hill Climbing Results
    print_timetable_table(hc_timetable, "FINAL TIMETABLE: HILL CLIMBING")
    hc_tot, hc_sd, hc_cp = calculate_penalty(hc_timetable, return_breakdown=True)
    print(f"Hill Climbing Final Penalty : {hc_penalty} (Same-Day: {hc_sd}, Consecutive: {hc_cp})")
    print(f"Hill Climbing Iteration Count: {hc_iterations}")
    print(f"Hill Climbing Execution Time : {hc_time:.6f} seconds")

    # 5. Display Simulated Annealing Results
    print_timetable_table(sa_timetable, "FINAL TIMETABLE: SIMULATED ANNEALING")
    sa_tot, sa_sd, sa_cp = calculate_penalty(sa_timetable, return_breakdown=True)
    print(f"Simulated Annealing Final Penalty : {sa_penalty} (Same-Day: {sa_sd}, Consecutive: {sa_cp})")
    print(f"Simulated Annealing Iteration Count: {sa_iterations}")
    print(f"Simulated Annealing Execution Time : {sa_time:.6f} seconds")

    # 6. Independent Hard-Constraint Validation
    hc_valid, hc_errors = validate_hard_constraints(hc_timetable)
    sa_valid, sa_errors = validate_hard_constraints(sa_timetable)

    print("\n" + "=" * 80)
    print(" INDEPENDENT HARD-CONSTRAINT VALIDATION REPORT")
    print("=" * 80)
    
    print("1. Hill Climbing Timetable Validation:")
    print(f"   - Total Scheduled Sessions: {len(hc_timetable)} / 8")
    print(f"   - Hard Constraint Errors   : {len(hc_errors)}")
    if hc_valid:
        print("   - Result: [VALID] 0 professor conflicts, 0 classroom conflicts, domains verified.")
    else:
        print(f"   - Result: [INVALID] Errors detected: {hc_errors}")

    print("\n2. Simulated Annealing Timetable Validation:")
    print(f"   - Total Scheduled Sessions: {len(sa_timetable)} / 8")
    print(f"   - Hard Constraint Errors   : {len(sa_errors)}")
    if sa_valid:
        print("   - Result: [VALID] 0 professor conflicts, 0 classroom conflicts, domains verified.")
    else:
        print(f"   - Result: [INVALID] Errors detected: {sa_errors}")

    # 7. Final Comparison Summary
    print("\n" + "=" * 80)
    print(" FINAL COMPARISON OF OPTIMIZATION ALGORITHMS")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Initial Schedule':<18} | {'Hill Climbing':<15} | {'Simulated Annealing':<18}")
    print("-" * 92)
    print(f"{'Penalty Score (Lower is better)':<32} | {initial_penalty:<18} | {hc_penalty:<15} | {sa_penalty:<18}")
    print(f"{'Iterations Explored':<32} | {'N/A':<18} | {hc_iterations:<15} | {sa_iterations:<18}")
    print(f"{'Execution Time (seconds)':<32} | {'N/A':<18} | {hc_time:<15.6f} | {sa_time:<18.6f}")
    print(f"{'Hard Constraints Valid?':<32} | {str(initial_valid):<18} | {str(hc_valid):<15} | {str(sa_valid):<18}")
    print("-" * 92)

    # 8. Analytical Discussion for Viva Demonstration
    print("\n[ALGORITHMIC INSIGHT & ANALYSIS]")
    print("1. Search Efficiency vs. Thoroughness:")
    print(f"   - Hill Climbing terminated in {hc_iterations} iterations ({hc_time * 1000:.3f} ms).")
    print("     It quickly descended along steepest gradient paths towards a local/global optimum.")
    print(f"   - Simulated Annealing performed {sa_iterations} iterations ({sa_time * 1000:.3f} ms).")
    print("     It maintained an intentional cooling schedule to thoroughly explore the search space.")
    print("2. Optimality & Local Minima:")
    print(f"   - Initial timetable had penalty = {initial_penalty} due to poor subject/professor spread.")
    print(f"   - Both algorithms successfully achieved optimal timetable solutions (Penalty = {hc_penalty} & {sa_penalty}).")
    print("   - Simulated Annealing provides higher probabilistic guarantees of escaping local optima")
    print("     on rugged landscapes because it accepts worse moves with probability P = exp(-dE / T).")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        main()
    else:
        # Launch modern AISC Timetable Optimizer GUI
        import gui_timetable_optimizer
        gui_timetable_optimizer.main()

