"""
================================================================================
AI Timetable Optimizer - GUI Application
Subject: Artificial Intelligence and Soft Computing (AISC) Practical
Algorithms: Hill Climbing & Simulated Annealing
GUI: Tkinter + Matplotlib
================================================================================
"""

import copy
import math
import random
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. CONSTANTS AND PROBLEM CONFIGURATION
# ==============================================================================

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_SLOTS = ["9–10 AM", "10–11 AM", "11 AM–12 PM", "1–2 PM"]
ROOMS = ["Room 101", "Room 102"]

SUBJECTS = [
    "Computer Networks",
    "AISC",
    "Database Systems",
    "Operating Systems"
]

PROFESSORS = [
    "Dr. Usha",
    "Dr. Pallavi",
    "Dr. Mayank",
    "Dr. Surbhi"
]

DEFAULT_SUBJECT_PROFESSOR = {
    "Computer Networks": "Dr. Usha",
    "AISC": "Dr. Pallavi",
    "Database Systems": "Dr. Mayank",
    "Operating Systems": "Dr. Surbhi"
}

# Undesirable time slot (post-lunch afternoon slot)
UNDESIRABLE_TIME_SLOT = "1–2 PM"

# Exact Penalty Weights as specified in requirements
WEIGHT_PROF_CLASH = 10        # 10 points per professor double-booking
WEIGHT_ROOM_CLASH = 8         # 8 points per room double-booking
WEIGHT_UNDESIRABLE_SLOT = 2   # 2 points per class scheduled in 1-2 PM

RANDOM_SEED = 42
MAX_ITERATIONS = 200


# ==============================================================================
# 2. CORE DOMAIN LOGIC: INITIAL TIMETABLE & PENALTY CALCULATION
# ==============================================================================

def generate_initial_timetable(seed=RANDOM_SEED):
    """
    Generates a reproducible initial timetable of 12 classes (3 per subject)
    using a fixed random seed.
    Contains realistic clashes and undesirable slots for demonstration.
    """
    random.seed(seed)
    timetable = []
    class_id = 1
    for subject, professor in DEFAULT_SUBJECT_PROFESSOR.items():
        for session_num in range(1, 4):
            timetable.append({
                "id": class_id,
                "subject": subject,
                "professor": professor,
                "session_num": session_num,
                "day": random.choice(DAYS),
                "time_slot": random.choice(TIME_SLOTS),
                "room": random.choice(ROOMS)
            })
            class_id += 1
    return timetable


def calculate_penalty(timetable):
    """
    Calculates total penalty and violations breakdown according to exact project weights:
    - Professor clash: 10 points (professor scheduled for >1 class at same day & time slot)
    - Room clash: 8 points (room scheduled for >1 class at same day & time slot)
    - Undesirable time slot (1-2 PM): 2 points per class

    Returns a dictionary of actual calculated metrics.
    """
    prof_clashes = 0
    room_clashes = 0
    undesirable_count = 0

    prof_slot_map = {}
    room_slot_map = {}

    for item in timetable:
        p_key = (item["professor"], item["day"], item["time_slot"])
        prof_slot_map[p_key] = prof_slot_map.get(p_key, 0) + 1

        r_key = (item["room"], item["day"], item["time_slot"])
        room_slot_map[r_key] = room_slot_map.get(r_key, 0) + 1

        if item["time_slot"] == UNDESIRABLE_TIME_SLOT:
            undesirable_count += 1

    # Clashes occur when count > 1
    for count in prof_slot_map.values():
        if count > 1:
            prof_clashes += (count - 1)

    for count in room_slot_map.values():
        if count > 1:
            room_clashes += (count - 1)

    total_penalty = (prof_clashes * WEIGHT_PROF_CLASH) + \
                    (room_clashes * WEIGHT_ROOM_CLASH) + \
                    (undesirable_count * WEIGHT_UNDESIRABLE_SLOT)

    total_violations = prof_clashes + room_clashes + undesirable_count

    return {
        "total_penalty": total_penalty,
        "total_violations": total_violations,
        "prof_clashes": prof_clashes,
        "room_clashes": room_clashes,
        "undesirable_count": undesirable_count
    }


def get_neighbor(timetable):
    """
    Generates a neighboring timetable by:
    - 75% probability: randomly reassigning a class's day, time slot, or room.
    - 25% probability: swapping the time-and-room slots of two classes.
    """
    neighbor = copy.deepcopy(timetable)
    idx1 = random.randint(0, len(neighbor) - 1)

    if random.random() < 0.75:
        mode = random.choice(["day", "time_slot", "room", "slot_room"])
        if mode == "day":
            neighbor[idx1]["day"] = random.choice(DAYS)
        elif mode == "time_slot":
            neighbor[idx1]["time_slot"] = random.choice(TIME_SLOTS)
        elif mode == "room":
            neighbor[idx1]["room"] = random.choice(ROOMS)
        else:
            neighbor[idx1]["day"] = random.choice(DAYS)
            neighbor[idx1]["time_slot"] = random.choice(TIME_SLOTS)
            neighbor[idx1]["room"] = random.choice(ROOMS)
    else:
        idx2 = random.randint(0, len(neighbor) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(neighbor) - 1)

        # Swap day, time_slot, and room
        neighbor[idx1]["day"], neighbor[idx2]["day"] = neighbor[idx2]["day"], neighbor[idx1]["day"]
        neighbor[idx1]["time_slot"], neighbor[idx2]["time_slot"] = neighbor[idx2]["time_slot"], neighbor[idx1]["time_slot"]
        neighbor[idx1]["room"], neighbor[idx2]["room"] = neighbor[idx2]["room"], neighbor[idx1]["room"]

    return neighbor


# ==============================================================================
# 3. ALGORITHM IMPLEMENTATIONS
# ==============================================================================

def run_hill_climbing(initial_timetable, max_iterations=MAX_ITERATIONS):
    """
    Hill Climbing Algorithm:
    - Starts from initial timetable.
    - Generates a neighbor at each iteration.
    - Strictly accepts neighbor only when it improves the penalty (n_penalty < cur_penalty).
    - Records penalty at each iteration and measures execution time.
    """
    start_time = time.perf_counter()
    current = copy.deepcopy(initial_timetable)
    current_metrics = calculate_penalty(current)
    current_penalty = current_metrics["total_penalty"]

    history = [current_penalty]

    for _ in range(max_iterations):
        neighbor = get_neighbor(current)
        n_metrics = calculate_penalty(neighbor)
        n_penalty = n_metrics["total_penalty"]

        # Strictly greedy acceptance criterion
        if n_penalty < current_penalty:
            current = neighbor
            current_penalty = n_penalty

        history.append(current_penalty)

    exec_time = time.perf_counter() - start_time
    final_metrics = calculate_penalty(current)

    return {
        "timetable": current,
        "final_penalty": current_penalty,
        "history": history,
        "execution_time": exec_time,
        "metrics": final_metrics
    }


def run_simulated_annealing(initial_timetable, initial_temp=30.0, cooling_rate=0.96, max_iterations=MAX_ITERATIONS):
    """
    Simulated Annealing Algorithm:
    - Starts from the same initial timetable.
    - Generates a neighbor at each iteration.
    - Improving moves (delta < 0) are always accepted.
    - Worse moves (delta >= 0) are accepted with probability P = exp(-delta / temperature).
    - Early in the search (high temperature), worse moves are frequently accepted.
    - As temperature cools, acceptance of worse moves decreases towards zero.
    - Tracks overall best timetable and records penalty at each iteration.
    """
    start_time = time.perf_counter()
    current = copy.deepcopy(initial_timetable)
    current_metrics = calculate_penalty(current)
    current_penalty = current_metrics["total_penalty"]

    best = copy.deepcopy(current)
    best_penalty = current_penalty

    history = [current_penalty]
    temp = initial_temp

    for _ in range(max_iterations):
        neighbor = get_neighbor(current)
        n_metrics = calculate_penalty(neighbor)
        n_penalty = n_metrics["total_penalty"]
        delta = n_penalty - current_penalty

        if delta < 0:
            current = neighbor
            current_penalty = n_penalty
            if current_penalty < best_penalty:
                best = copy.deepcopy(current)
                best_penalty = current_penalty
        else:
            # Boltzmann acceptance criterion
            prob = math.exp(-delta / max(temp, 1e-6))
            if random.random() < prob:
                current = neighbor
                current_penalty = n_penalty

        history.append(current_penalty)
        temp *= cooling_rate

    exec_time = time.perf_counter() - start_time
    best_metrics = calculate_penalty(best)

    return {
        "timetable": best,
        "final_penalty": best_penalty,
        "history": history,
        "execution_time": exec_time,
        "metrics": best_metrics
    }


# ==============================================================================
# 4. GUI IMPLEMENTATION (TKINTER + MATPLOTLIB)
# ==============================================================================

class TimetableOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Timetable Optimizer - Hill Climbing & Simulated Annealing")
        self.root.geometry("1260x820")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#f8f9fa")

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#f8f9fa", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#f8f9fa", foreground="#1a202c")
        self.style.configure("Subheader.TLabel", font=("Segoe UI", 10), background="#f8f9fa", foreground="#4a5568")
        self.style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#2b6cb0")

        # Application state
        self.initial_timetable = generate_initial_timetable(RANDOM_SEED)
        self.current_timetable = copy.deepcopy(self.initial_timetable)
        self.hc_result = None
        self.sa_result = None
        self.active_view = "current"  # "current", "hc", "sa"

        self.selected_class_id = tk.IntVar(value=1)

        self._build_ui()
        self._load_selected_class_into_editors()
        self._refresh_all_views()

    def _build_ui(self):
        # 1. Top Header Bar
        header_frame = tk.Frame(self.root, bg="#ffffff", pady=10, padx=20, relief="groove", bd=1)
        header_frame.pack(fill="x", side="top")

        title_lbl = tk.Label(header_frame, text="AI Timetable Optimizer", font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#2d3748")
        title_lbl.pack(side="left")

        subtitle_lbl = tk.Label(
            header_frame,
            text="AISC Practical Project | Algorithms: Hill Climbing & Simulated Annealing",
            font=("Segoe UI", 10), bg="#ffffff", fg="#718096"
        )
        subtitle_lbl.pack(side="left", padx=15, pady=4)

        # 2. Main Content Container (2 Columns)
        main_container = tk.Frame(self.root, bg="#f8f9fa", padx=15, pady=10)
        main_container.pack(fill="both", expand=True)

        # Left Column: Timetable Grid & Editor (60% width)
        left_col = tk.Frame(main_container, bg="#f8f9fa")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right Column: Penalties, Controls, Graph, Comparison Table (40% width)
        right_col = tk.Frame(main_container, bg="#f8f9fa", width=520)
        right_col.pack(side="right", fill="both", expand=False)

        self._build_left_column(left_col)
        self._build_right_column(right_col)

    def _build_left_column(self, parent):
        # Card 1: Weekly Timetable Grid
        grid_card = tk.LabelFrame(parent, text=" Weekly Timetable Schedule ", font=("Segoe UI", 11, "bold"), bg="#ffffff", padx=10, pady=10)
        grid_card.pack(fill="both", expand=True, pady=(0, 10))

        # View selector buttons (Switch between Current, HC Best, SA Best)
        view_bar = tk.Frame(grid_card, bg="#ffffff")
        view_bar.pack(fill="x", pady=(0, 8))

        tk.Label(view_bar, text="Displaying:", font=("Segoe UI", 9, "bold"), bg="#ffffff").pack(side="left", padx=(0, 5))
        self.view_var = tk.StringVar(value="current")
        self.rb_curr = ttk.Radiobutton(view_bar, text="Editable Timetable", variable=self.view_var, value="current", command=self._on_view_changed)
        self.rb_curr.pack(side="left", padx=5)
        self.rb_hc = ttk.Radiobutton(view_bar, text="Hill Climbing Best", variable=self.view_var, value="hc", command=self._on_view_changed)
        self.rb_hc.pack(side="left", padx=5)
        self.rb_sa = ttk.Radiobutton(view_bar, text="Simulated Annealing Best", variable=self.view_var, value="sa", command=self._on_view_changed)
        self.rb_sa.pack(side="left", padx=5)

        # Weekly Grid Container
        self.grid_container = tk.Frame(grid_card, bg="#cbd5e0", bd=1, relief="solid")
        self.grid_container.pack(fill="both", expand=True)

        self._render_timetable_grid()

        # Card 2: Interactive Class Editor
        editor_card = tk.LabelFrame(parent, text=" Edit Scheduled Class ", font=("Segoe UI", 11, "bold"), bg="#ffffff", padx=12, pady=10)
        editor_card.pack(fill="x")

        # Row 1: Select class
        sel_row = tk.Frame(editor_card, bg="#ffffff")
        sel_row.pack(fill="x", pady=(0, 8))

        tk.Label(sel_row, text="Select Class:", font=("Segoe UI", 9, "bold"), bg="#ffffff").pack(side="left", padx=(0, 5))
        self.class_selector_cb = ttk.Combobox(sel_row, state="readonly", width=45)
        self.class_selector_cb.pack(side="left", padx=5)
        self.class_selector_cb.bind("<<ComboboxSelected>>", self._on_class_selected_from_combobox)

        # Row 2: Dropdowns for Subject, Professor, Day, Slot, Room
        form_row = tk.Frame(editor_card, bg="#ffffff")
        form_row.pack(fill="x", pady=(0, 8))

        # Subject
        f_sub = tk.Frame(form_row, bg="#ffffff")
        f_sub.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f_sub, text="Subject:", font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568").pack(anchor="w")
        self.subject_cb = ttk.Combobox(f_sub, values=SUBJECTS, state="readonly", width=18)
        self.subject_cb.pack(fill="x")
        self.subject_cb.bind("<<ComboboxSelected>>", self._on_subject_changed)

        # Professor
        f_prof = tk.Frame(form_row, bg="#ffffff")
        f_prof.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f_prof, text="Professor:", font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568").pack(anchor="w")
        self.prof_cb = ttk.Combobox(f_prof, values=PROFESSORS, state="readonly", width=14)
        self.prof_cb.pack(fill="x")

        # Day
        f_day = tk.Frame(form_row, bg="#ffffff")
        f_day.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f_day, text="Day:", font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568").pack(anchor="w")
        self.day_cb = ttk.Combobox(f_day, values=DAYS, state="readonly", width=12)
        self.day_cb.pack(fill="x")

        # Time Slot
        f_slot = tk.Frame(form_row, bg="#ffffff")
        f_slot.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f_slot, text="Time Slot:", font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568").pack(anchor="w")
        self.slot_cb = ttk.Combobox(f_slot, values=TIME_SLOTS, state="readonly", width=14)
        self.slot_cb.pack(fill="x")

        # Room
        f_room = tk.Frame(form_row, bg="#ffffff")
        f_room.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f_room, text="Room:", font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568").pack(anchor="w")
        self.room_cb = ttk.Combobox(f_room, values=ROOMS, state="readonly", width=12)
        self.room_cb.pack(fill="x")

        # Row 3: Action Buttons
        btn_row = tk.Frame(editor_card, bg="#ffffff")
        btn_row.pack(fill="x")

        self.apply_btn = tk.Button(
            btn_row, text="Apply Changes to Class", bg="#2b6cb0", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._apply_class_edits
        )
        self.apply_btn.pack(side="left", padx=(0, 8))

        self.recalc_btn = tk.Button(
            btn_row, text="Recalculate Penalty", bg="#4a5568", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._on_recalculate_clicked
        )
        self.recalc_btn.pack(side="left", padx=8)

        self.reset_btn = tk.Button(
            btn_row, text="Reset to Initial State", bg="#e2e8f0", fg="#2d3748",
            font=("Segoe UI", 9), relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._reset_to_initial
        )
        self.reset_btn.pack(side="right")

    def _build_right_column(self, parent):
        # 1. Penalty Score & Violation Breakdown Card
        penalty_card = tk.LabelFrame(parent, text=" Penalty & Constraint Status ", font=("Segoe UI", 11, "bold"), bg="#ffffff", padx=12, pady=10)
        penalty_card.pack(fill="x", pady=(0, 10))

        # Top Penalty Summary Banner
        p_banner = tk.Frame(penalty_card, bg="#ebf8ff", bd=1, relief="solid", padx=10, pady=6)
        p_banner.pack(fill="x", pady=(0, 8))

        self.total_penalty_lbl = tk.Label(p_banner, text="Total Penalty: 30 pts", font=("Segoe UI", 13, "bold"), bg="#ebf8ff", fg="#2c5282")
        self.total_penalty_lbl.pack(side="left")

        self.total_violations_lbl = tk.Label(p_banner, text="Violations: 5", font=("Segoe UI", 11, "bold"), bg="#ebf8ff", fg="#c53030")
        self.total_violations_lbl.pack(side="right")

        # Detailed Breakdown Labels
        breakdown_frame = tk.Frame(penalty_card, bg="#ffffff")
        breakdown_frame.pack(fill="x")

        self.prof_clash_lbl = tk.Label(breakdown_frame, text="• Professor Clashes (10 pts each): 1 (10 pts)", font=("Segoe UI", 9), bg="#ffffff", fg="#4a5568")
        self.prof_clash_lbl.pack(anchor="w")

        self.room_clash_lbl = tk.Label(breakdown_frame, text="• Room Clashes (8 pts each): 2 (16 pts)", font=("Segoe UI", 9), bg="#ffffff", fg="#4a5568")
        self.room_clash_lbl.pack(anchor="w")

        self.undesirable_lbl = tk.Label(breakdown_frame, text="• Undesirable Slot 1–2 PM (2 pts each): 2 (4 pts)", font=("Segoe UI", 9), bg="#ffffff", fg="#4a5568")
        self.undesirable_lbl.pack(anchor="w")

        # 2. Algorithm Execution Controls
        control_card = tk.Frame(parent, bg="#ffffff", padx=10, pady=10, relief="solid", bd=1)
        control_card.pack(fill="x", pady=(0, 10))

        self.run_btn = tk.Button(
            control_card, text="▶  Run Both Algorithms (Hill Climbing & Simulated Annealing)",
            bg="#2f855a", fg="#ffffff", font=("Segoe UI", 11, "bold"),
            relief="flat", pady=7, cursor="hand2", command=self._run_algorithms
        )
        self.run_btn.pack(fill="x")

        self.load_best_btn = tk.Button(
            control_card, text="📥  Load Best Solution into Editable Timetable",
            bg="#edf2f7", fg="#2d3748", font=("Segoe UI", 9, "bold"),
            relief="solid", bd=1, pady=4, cursor="hand2", command=self._load_best_into_editable
        )
        self.load_best_btn.pack(fill="x", pady=(6, 0))

        self.best_status_lbl = tk.Label(
            control_card, text="Status: Ready. Click 'Run Both Algorithms' to optimize.",
            font=("Segoe UI", 9, "italic"), bg="#ffffff", fg="#718096"
        )
        self.best_status_lbl.pack(pady=(6, 0))


        # 3. Small Comparison Table
        comp_card = tk.LabelFrame(parent, text=" Algorithm Comparison Table ", font=("Segoe UI", 11, "bold"), bg="#ffffff", padx=10, pady=8)
        comp_card.pack(fill="x", pady=(0, 10))

        cols = ("Metric", "Initial", "Hill Climbing", "Simulated Annealing")
        self.comp_tree = ttk.Treeview(comp_card, columns=cols, show="headings", height=5)
        for col in cols:
            self.comp_tree.heading(col, text=col)

        self.comp_tree.column("Metric", width=140, anchor="w")
        self.comp_tree.column("Initial", width=80, anchor="center")
        self.comp_tree.column("Hill Climbing", width=110, anchor="center")
        self.comp_tree.column("Simulated Annealing", width=130, anchor="center")
        self.comp_tree.pack(fill="x")

        # 4. Embedded Matplotlib Graph
        graph_card = tk.LabelFrame(parent, text=" Penalty vs. Iterations Convergence Graph ", font=("Segoe UI", 11, "bold"), bg="#ffffff", padx=8, pady=6)
        graph_card.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(5.2, 2.7), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#fafafa")
        self.ax.set_title("Penalty vs. Iterations", fontsize=10, fontweight="bold", pad=8)
        self.ax.set_xlabel("Iterations", fontsize=9)
        self.ax.set_ylabel("Penalty Score", fontsize=9)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Draw empty initial placeholder graph
        self._plot_graph([], [])

    # ==========================================================================
    # 5. GRID RENDERING & SELECTION LOGIC
    # ==========================================================================

    def _render_timetable_grid(self):
        """
        Renders the weekly grid (5 days as columns, 4 time slots as rows).
        Highlights clashes and allows selecting classes directly.
        """
        # Clear existing widgets
        for widget in self.grid_container.winfo_children():
            widget.destroy()

        # Decide which timetable data to display
        if self.active_view == "hc" and self.hc_result is not None:
            active_data = self.hc_result["timetable"]
        elif self.active_view == "sa" and self.sa_result is not None:
            active_data = self.sa_result["timetable"]
        else:
            active_data = self.current_timetable

        # Header Row: Time Slot, Monday, Tuesday, Wednesday, Thursday, Friday
        headers = ["Time Slot"] + DAYS
        for col_idx, h in enumerate(headers):
            lbl = tk.Label(
                self.grid_container, text=h, font=("Segoe UI", 9, "bold"),
                bg="#edf2f7", fg="#2d3748", relief="solid", bd=1, padx=4, pady=4
            )
            lbl.grid(row=0, column=col_idx, sticky="nsew")
            self.grid_container.columnconfigure(col_idx, weight=1)

        # Build lookup table: (day, time_slot) -> list of classes
        slot_map = {}
        for item in active_data:
            key = (item["day"], item["time_slot"])
            if key not in slot_map:
                slot_map[key] = []
            slot_map[key].append(item)

        # Check for conflicts in active data
        prof_counts = {}
        room_counts = {}
        for item in active_data:
            pk = (item["professor"], item["day"], item["time_slot"])
            rk = (item["room"], item["day"], item["time_slot"])
            prof_counts[pk] = prof_counts.get(pk, 0) + 1
            room_counts[rk] = room_counts.get(rk, 0) + 1

        # Populate rows
        for row_idx, slot in enumerate(TIME_SLOTS, start=1):
            # Time Slot Column Label
            is_undesirable = (slot == UNDESIRABLE_TIME_SLOT)
            slot_bg = "#feebc8" if is_undesirable else "#edf2f7"
            slot_text = slot + ("\n(Undesirable)" if is_undesirable else "")

            s_lbl = tk.Label(
                self.grid_container, text=slot_text, font=("Segoe UI", 8, "bold" if is_undesirable else "normal"),
                bg=slot_bg, fg="#742a2a" if is_undesirable else "#2d3748",
                relief="solid", bd=1, padx=2, pady=4
            )
            s_lbl.grid(row=row_idx, column=0, sticky="nsew")
            self.grid_container.rowconfigure(row_idx, weight=1)

            # Days columns
            for col_idx, day in enumerate(DAYS, start=1):
                cell_frame = tk.Frame(self.grid_container, bg="#ffffff", relief="solid", bd=1, padx=2, pady=2)
                cell_frame.grid(row=row_idx, column=col_idx, sticky="nsew")

                classes_in_cell = slot_map.get((day, slot), [])

                if not classes_in_cell:
                    empty_lbl = tk.Label(cell_frame, text="—", font=("Segoe UI", 8), bg="#ffffff", fg="#a0aec0")
                    empty_lbl.pack(expand=True)
                else:
                    for cls in classes_in_cell:
                        pk = (cls["professor"], cls["day"], cls["time_slot"])
                        rk = (cls["room"], cls["day"], cls["time_slot"])
                        has_prof_clash = prof_counts[pk] > 1
                        has_room_clash = room_counts[rk] > 1

                        # Color tag: Red for clashes, Green for clean
                        if has_prof_clash or has_room_clash:
                            tag_bg = "#fed7d7"
                            tag_fg = "#9b2c2c"
                            clash_note = " [CLASH]"
                        else:
                            tag_bg = "#e6fffa"
                            tag_fg = "#234e52"
                            clash_note = ""

                        # Short subject name
                        subj_short = {
                            "Computer Networks": "CN",
                            "AISC": "AISC",
                            "Database Systems": "DBMS",
                            "Operating Systems": "OS"
                        }.get(cls["subject"], cls["subject"])

                        room_short = "R101" if "101" in cls["room"] else "R102"
                        cls_text = f"[{room_short}] {subj_short} ({cls['professor'].split()[-1]}){clash_note}"

                        # Highlight if this class is currently selected in editor
                        is_sel = (cls["id"] == self.selected_class_id.get())
                        b_width = 2 if is_sel else 1
                        b_color = "#3182ce" if is_sel else tag_bg

                        btn = tk.Button(
                            cell_frame, text=cls_text, font=("Segoe UI", 8, "bold" if is_sel else "normal"),
                            bg=tag_bg, fg=tag_fg, relief="solid", bd=b_width,
                            cursor="hand2", padx=2, pady=1, anchor="w",
                            command=lambda cid=cls["id"]: self._select_class_by_id(cid)
                        )
                        btn.pack(fill="x", pady=1)

    def _select_class_by_id(self, class_id):
        self.selected_class_id.set(class_id)
        self._load_selected_class_into_editors()
        self._render_timetable_grid()

    def _on_class_selected_from_combobox(self, event=None):
        selected_text = self.class_selector_cb.get()
        if selected_text:
            cid = int(selected_text.split(":")[0].replace("Class", "").strip())
            self.selected_class_id.set(cid)
            self._load_selected_class_into_editors()
            self._render_timetable_grid()

    def _load_selected_class_into_editors(self):
        """Populates the dropdowns with current values of the selected class."""
        cid = self.selected_class_id.get()
        target = next((item for item in self.current_timetable if item["id"] == cid), None)
        if target:
            self.subject_cb.set(target["subject"])
            self.prof_cb.set(target["professor"])
            self.day_cb.set(target["day"])
            self.slot_cb.set(target["time_slot"])
            self.room_cb.set(target["room"])

        # Update combobox dropdown choices list
        choices = []
        for item in self.current_timetable:
            choices.append(f"Class {item['id']}: {item['subject']} | {item['day']} {item['time_slot']} | {item['room']} ({item['professor']})")
        self.class_selector_cb["values"] = choices

        # Match current index
        idx = next((i for i, c in enumerate(self.current_timetable) if c["id"] == cid), 0)
        self.class_selector_cb.current(idx)

    def _on_subject_changed(self, event=None):
        # Automatically select the assigned professor for the subject
        sub = self.subject_cb.get()
        if sub in DEFAULT_SUBJECT_PROFESSOR:
            self.prof_cb.set(DEFAULT_SUBJECT_PROFESSOR[sub])

    def _apply_class_edits(self):
        """Updates the timetable entry with values from the editor dropdowns."""
        cid = self.selected_class_id.get()
        target = next((item for item in self.current_timetable if item["id"] == cid), None)
        if not target:
            return

        # Validate entries against allowed domains
        new_sub = self.subject_cb.get()
        new_prof = self.prof_cb.get()
        new_day = self.day_cb.get()
        new_slot = self.slot_cb.get()
        new_room = self.room_cb.get()

        if new_sub not in SUBJECTS or new_prof not in PROFESSORS or new_day not in DAYS or new_slot not in TIME_SLOTS or new_room not in ROOMS:
            messagebox.showerror("Invalid Input", "Selected values must belong to valid domains (unknown subject/professor/day/room).")
            return

        target["subject"] = new_sub
        target["professor"] = new_prof
        target["day"] = new_day
        target["time_slot"] = new_slot
        target["room"] = new_room

        # Switch back to editable view if we were viewing HC/SA
        self.view_var.set("current")
        self.active_view = "current"

        self._refresh_all_views()
        self._load_selected_class_into_editors()

    def _on_recalculate_clicked(self):
        """Explicitly re-evaluates and displays the current penalty and violations."""
        self._refresh_all_views()

    def _reset_to_initial(self):
        """Resets the timetable to the initial reproducible seed 42 state."""
        self.current_timetable = copy.deepcopy(self.initial_timetable)
        self.view_var.set("current")
        self.active_view = "current"
        self._load_selected_class_into_editors()
        self._refresh_all_views()

    def _on_view_changed(self):
        self.active_view = self.view_var.get()
        self._render_timetable_grid()
        self._refresh_penalty_display()

    # ==========================================================================
    # 6. PENALTY DISPLAY & ALGORITHM EXECUTION
    # ==========================================================================

    def _refresh_all_views(self):
        self._render_timetable_grid()
        self._refresh_penalty_display()
        self._update_comparison_table()

    def _refresh_penalty_display(self):
        # Choose data
        if self.active_view == "hc" and self.hc_result is not None:
            data = self.hc_result["timetable"]
            source = "Hill Climbing Best"
        elif self.active_view == "sa" and self.sa_result is not None:
            data = self.sa_result["timetable"]
            source = "Simulated Annealing Best"
        else:
            data = self.current_timetable
            source = "Editable Timetable"

        metrics = calculate_penalty(data)
        tot = metrics["total_penalty"]
        viol = metrics["total_violations"]

        self.total_penalty_lbl.config(text=f"Total Penalty: {tot} pts ({source})")
        self.total_violations_lbl.config(text=f"Violations: {viol}")

        self.prof_clash_lbl.config(
            text=f"• Professor Clashes (10 pts each): {metrics['prof_clashes']} ({metrics['prof_clashes'] * WEIGHT_PROF_CLASH} pts)"
        )
        self.room_clash_lbl.config(
            text=f"• Room Clashes (8 pts each): {metrics['room_clashes']} ({metrics['room_clashes'] * WEIGHT_ROOM_CLASH} pts)"
        )
        self.undesirable_lbl.config(
            text=f"• Undesirable Slot 1–2 PM (2 pts each): {metrics['undesirable_count']} ({metrics['undesirable_count'] * WEIGHT_UNDESIRABLE_SLOT} pts)"
        )

    def _run_algorithms(self):
        """Runs both Hill Climbing and Simulated Annealing from the current timetable."""
        # Use fixed seed for exact reproducible comparison
        random.seed(RANDOM_SEED)
        self.hc_result = run_hill_climbing(self.current_timetable, max_iterations=MAX_ITERATIONS)

        random.seed(RANDOM_SEED)
        self.sa_result = run_simulated_annealing(self.current_timetable, initial_temp=30.0, cooling_rate=0.96, max_iterations=MAX_ITERATIONS)

        # Update graph
        self._plot_graph(self.hc_result["history"], self.sa_result["history"])

        # Update comparison table
        self._update_comparison_table()

        # Update status message with best timetable found
        hc_pen = self.hc_result["final_penalty"]
        sa_pen = self.sa_result["final_penalty"]

        if hc_pen < sa_pen:
            best_algo = "Hill Climbing"
            best_pen = hc_pen
            self.view_var.set("hc")
            self.active_view = "hc"
        elif sa_pen < hc_pen:
            best_algo = "Simulated Annealing"
            best_pen = sa_pen
            self.view_var.set("sa")
            self.active_view = "sa"
        else:
            best_algo = "Both algorithms tied"
            best_pen = hc_pen
            self.view_var.set("sa")
            self.active_view = "sa"

        self.best_status_lbl.config(
            text=f"Optimization Complete! Best Solution: {best_algo} (Penalty: {best_pen} pts).",
            fg="#22543d", font=("Segoe UI", 9, "bold")
        )

        self._refresh_all_views()

    def _load_best_into_editable(self):
        """Loads the best algorithm timetable into the editable timetable so the user can interact with it."""
        if not self.hc_result and not self.sa_result:
            messagebox.showinfo("Run Algorithms First", "Please click 'Run Both Algorithms' first to generate optimized solutions.")
            return

        hc_pen = self.hc_result["final_penalty"] if self.hc_result else float('inf')
        sa_pen = self.sa_result["final_penalty"] if self.sa_result else float('inf')

        if hc_pen <= sa_pen and self.hc_result:
            best_data = self.hc_result["timetable"]
            best_name = "Hill Climbing"
        else:
            best_data = self.sa_result["timetable"]
            best_name = "Simulated Annealing"

        self.current_timetable = copy.deepcopy(best_data)
        self.view_var.set("current")
        self.active_view = "current"
        self._load_selected_class_into_editors()
        self._refresh_all_views()
        messagebox.showinfo("Loaded Best Solution", f"Loaded {best_name}'s best solution into the editable timetable grid!")

    def _update_comparison_table(self):

        """Updates the small comparison table with genuine runtime data."""
        # Clear existing items
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)

        init_m = calculate_penalty(self.current_timetable)
        init_pen = init_m["total_penalty"]

        if self.hc_result and self.sa_result:
            hc_pen = self.hc_result["final_penalty"]
            sa_pen = self.sa_result["final_penalty"]

            hc_viol = self.hc_result["metrics"]["total_violations"]
            sa_viol = self.sa_result["metrics"]["total_violations"]

            # Improvement percentage formula: ((initial - final) / initial) * 100
            hc_imp = ((init_pen - hc_pen) / max(init_pen, 1)) * 100
            sa_imp = ((init_pen - sa_pen) / max(init_pen, 1)) * 100

            hc_time_str = f"{self.hc_result['execution_time'] * 1000:.2f} ms"
            sa_time_str = f"{self.sa_result['execution_time'] * 1000:.2f} ms"

            self.comp_tree.insert("", "end", values=("Final Penalty", f"{init_pen} pts", f"{hc_pen} pts", f"{sa_pen} pts"))
            self.comp_tree.insert("", "end", values=("Improvement (%)", "—", f"{hc_imp:.1f}%", f"{sa_imp:.1f}%"))
            self.comp_tree.insert("", "end", values=("Violations Left", f"{init_m['total_violations']}", f"{hc_viol}", f"{sa_viol}"))
            self.comp_tree.insert("", "end", values=("Execution Time", "—", hc_time_str, sa_time_str))
            
            hc_win = "Optimal" if hc_pen == 0 else ("Winner" if hc_pen < sa_pen else "Suboptimal")
            sa_win = "Optimal" if sa_pen == 0 else ("Winner" if sa_pen < hc_pen else "Suboptimal")
            self.comp_tree.insert("", "end", values=("Status", "Baseline", hc_win, sa_win))
        else:
            self.comp_tree.insert("", "end", values=("Final Penalty", f"{init_pen} pts", "Not Run", "Not Run"))
            self.comp_tree.insert("", "end", values=("Improvement (%)", "—", "—", "—"))
            self.comp_tree.insert("", "end", values=("Violations Left", f"{init_m['total_violations']}", "—", "—"))
            self.comp_tree.insert("", "end", values=("Execution Time", "—", "—", "—"))
            self.comp_tree.insert("", "end", values=("Status", "Baseline", "—", "—"))

    def _plot_graph(self, hc_history, sa_history):
        """Plots a single line graph comparing Hill Climbing and Simulated Annealing."""
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.set_title("Penalty vs. Iterations", fontsize=10, fontweight="bold", pad=8)
        self.ax.set_xlabel("Iterations", fontsize=9)
        self.ax.set_ylabel("Penalty Score", fontsize=9)
        self.ax.grid(True, linestyle="--", alpha=0.5)

        if hc_history and sa_history:
            iterations = list(range(len(hc_history)))
            self.ax.plot(iterations, hc_history, label=f"Hill Climbing (Final: {hc_history[-1]})", color="#2b6cb0", linewidth=2.0)
            self.ax.plot(iterations, sa_history, label=f"Simulated Annealing (Final: {sa_history[-1]})", color="#c53030", linewidth=2.0, linestyle="-")
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            self.ax.set_ylim(bottom=-1)
        else:
            self.ax.text(0.5, 0.5, "Click 'Run Both Algorithms' to view\nconvergence comparison graph",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, color="#a0aec0", fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()


# ==============================================================================
# 7. MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    root = tk.Tk()
    app = TimetableOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
