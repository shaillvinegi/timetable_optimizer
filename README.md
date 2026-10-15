# AI Timetable Optimizer - Python GUI Practical

**Subject:** Artificial Intelligence and Soft Computing (AISC) Practical  
**GUI Framework:** Tkinter (Python Standard Library)  
**Graphing:** Matplotlib (`FigureCanvasTkAgg` embedded directly in GUI)  
**Main Files:**
- [`gui_timetable_optimizer.py`](gui_timetable_optimizer.py) — Complete Tkinter & Matplotlib GUI application
- [`timetable_optimizer.py`](timetable_optimizer.py) — Universal entry point (launches GUI by default; supports `--cli` mode)

---

## 1. Project Overview & Aim
This project provides a clean, single-screen desktop GUI application to solve, visualize, and interactively optimize a college weekly timetable using two core AI search algorithms:
1. **Hill Climbing** (Greedy Local Search)
2. **Simulated Annealing** (Probabilistic Metaheuristic with Cooling Schedule)

The interface is designed specifically for college practicals and viva demonstrations: minimal, clean, white background, zero web frameworks or databases, and fully interactive.

---

## 2. Timetable Problem Specification

### Dataset
- **Working Days (5):** Monday, Tuesday, Wednesday, Thursday, Friday
- **Time Slots (4 per day):**
  - `9–10 AM`
  - `10–11 AM`
  - `11 AM–12 PM`
  - `1–2 PM` *(Designated as the undesirable post-lunch slot)*
- **Classrooms (2):** Room 101 and Room 102
- **Subjects and Faculty:**
  - **Computer Networks (CN)** — Dr. Usha
  - **AISC** — Dr. Pallavi
  - **Database Systems (DBMS)** — Dr. Mayank
  - **Operating Systems (OS)** — Dr. Surbhi
- **Required Classes:** Exactly **3 classes per week per subject** ($4 \times 3 = 12$ total classes).
- **Reproducible Baseline:** Generated with a fixed random seed (`seed = 42`) ensuring identical initial clashes and scores for every demonstration.

---

## 3. Penalty Function & Violation Weights

The penalty function quantifies the severity of schedule flaws. Lower scores represent better schedules (optimal score is **0**).

$$\text{Total Penalty} = (\text{Prof Clashes} \times 10) + (\text{Room Clashes} \times 8) + (\text{Undesirable Slots} \times 2)$$

| Violation Type | Weight | Description | Viva Explanation |
| :--- | :---: | :--- | :--- |
| **Professor Clash** | **10 pts** | Same professor scheduled for $>1$ class at the same (Day, Time Slot). | A professor physically cannot lecture in two rooms simultaneously. Highest penalty weight. |
| **Room Clash** | **8 pts** | Same room booked for $>1$ class at the same (Day, Time Slot). | A physical classroom cannot hold two different lectures at once. |
| **Undesirable Slot** | **2 pts** | Class scheduled in the post-lunch `1–2 PM` slot. | Suboptimal learning period due to post-lunch student fatigue. Soft preference to keep this slot free. |

### Baseline Initial State (Seed = 42):
- **1 Professor Clash:** $1 \times 10 = 10\text{ pts}$
- **2 Room Clashes:** $2 \times 8 = 16\text{ pts}$
- **2 Undesirable Slots:** $2 \times 2 = 4\text{ pts}$
- **Initial Total Penalty:** **30 points** (Total violations = **5**)

---

## 4. Key GUI Features

Everything is integrated on **one single clean screen**:

1. **Weekly Schedule Grid (Left Panel):**
   - Displays all 5 days (Monday to Friday) across 4 time slots.
   - Shows room assignment, subject short code, professor, and conflict tags.
   - Cells with conflicts are clearly highlighted in **soft red** with a `[CLASH]` badge.
   - Clicking any class directly inside a cell selects it for editing.
   - View selector radio buttons allow toggling between:
     - *Editable Timetable*
     - *Hill Climbing Best Result*
     - *Simulated Annealing Best Result*

2. **Interactive Class Editor (Bottom Left):**
   - Select any scheduled class (1 to 12) from the dropdown or by clicking the grid.
   - Edit **Subject**, **Professor**, **Day**, **Time Slot**, or **Room** using native dropdowns.
   - **Apply Changes to Class Button:** Updates the schedule, re-renders the grid, and recalculates the penalty and violations immediately.
   - **Recalculate Penalty Button:** Re-evaluates the entire schedule and updates the score breakdown.
   - **Reset to Initial State Button:** Instantly restores the baseline seed 42 timetable.

3. **Current Penalty & Constraint Card (Top Right):**
   - Displays real-time **Total Penalty** and **Violation Count**.
   - Details exact counts and point contributions for professor clashes, room clashes, and undesirable slots.

4. **Algorithm Execution & Comparison (Middle Right):**
   - **▶ Run Both Algorithms Button:** Executes Hill Climbing and Simulated Annealing from the exact current schedule.
   - **📥 Load Best Solution into Editable Timetable Button:** Copies the optimal 0-penalty timetable directly into the editable grid for inspection.
   - **Comparison Table:** Displays Initial Penalty, Final Penalty, Improvement (%), Violations Left, Execution Time (in ms), and Status for both algorithms.

5. **Embedded Convergence Graph (Bottom Right):**
   - Single line graph comparing **Penalty vs. Iterations**.
   - **Blue line:** Hill Climbing (strictly monotonic descent).
   - **Red line:** Simulated Annealing (early probabilistic fluctuations settling down as temperature cools).
   - Generated using Matplotlib embedded in Tkinter (`FigureCanvasTkAgg`).

---

## 5. How to Run the Program in VS Code

### Prerequisites
- Python 3.8 or higher.
- `matplotlib` installed in your Python environment:
  ```bash
  pip install matplotlib
  ```
  *(Tkinter comes built-in with standard Python on Windows).*

### Steps:
1. Open the project folder in VS Code (`File > Open Folder...`).
2. Open the integrated terminal (``Ctrl + ` ``).
3. Run the application:
   ```bash
   python gui_timetable_optimizer.py
   ```
   *(Alternatively, `python timetable_optimizer.py` also launches the GUI automatically).*
4. The desktop GUI window will appear immediately.

---

## 6. How the Graph Works (Viva Explanation)

The Matplotlib graph plots the penalty score at each iteration ($0$ to $200$):
- **Hill Climbing Curve (Blue):**
  - Moves strictly downhill.
  - At any step, a neighbor is accepted **only if** $\text{Penalty}_{\text{new}} < \text{Penalty}_{\text{current}}$.
  - The curve is monotonically non-increasing and flattens out as soon as an optimum is reached.
- **Simulated Annealing Curve (Red):**
  - Accepts improvements unconditionally ($\Delta E < 0$).
  - When a worse neighbor is proposed ($\Delta E \ge 0$), it is accepted with probability:
    $$P = \exp\left(-\frac{\Delta E}{T}\right)$$
  - Early in the run (high temperature $T = 30.0$), worse moves are accepted, causing visible **upward and downward fluctuations** on the graph.
  - As temperature cools geometrically ($T \leftarrow T \times 0.96$), $P \to 0$, dampening the fluctuations and converging smoothly to a zero-penalty schedule.

---

## 7. 10 Simple Viva Questions & Answers

### Q1. What is the difference between Hill Climbing and Simulated Annealing?
**Answer:** Hill Climbing is a purely greedy search that only accepts moves that immediately decrease the penalty. Simulated Annealing is a metaheuristic that always accepts improving moves, but also accepts worse moves with a probability that decreases as the system cools down, allowing it to escape local optima.

### Q2. How is the penalty calculated in this project?
**Answer:** The penalty is calculated dynamically using:
$$\text{Penalty} = (\text{Professor Clashes} \times 10) + (\text{Room Clashes} \times 8) + (\text{Undesirable Slots} \times 2)$$
A professor double-booking costs 10 points, a room double-booking costs 8 points, and using the 1–2 PM post-lunch slot costs 2 points.

### Q3. Why does a professor clash have a higher penalty than an undesirable time slot?
**Answer:** A professor clash is a hard operational conflict—a person physically cannot be in two classrooms at the same time. An undesirable time slot (1–2 PM) is merely a preference to avoid post-lunch fatigue. Therefore, the clash penalty (10 pts) is heavily prioritized over preference (2 pts).

### Q4. How is a "neighboring timetable" generated in your code?
**Answer:** A neighbor is generated by either randomly reassigning one class's day, time slot, or room (75% probability), or swapping the time-and-room assignments of two classes (25% probability).

### Q5. What is the acceptance probability formula in Simulated Annealing?
**Answer:** For a non-improving move where $\Delta E = \text{new\_penalty} - \text{current\_penalty} > 0$:
$$P = \exp\left(-\frac{\Delta E}{T}\right)$$
where $T$ is the current temperature. A random number $r \in [0, 1)$ is drawn; if $r < P$, the worse timetable is accepted.

### Q6. What does the cooling schedule do?
**Answer:** The cooling schedule gradually reduces the temperature using geometric cooling:
$$T_{k+1} = T_k \times \alpha \quad (\text{with } \alpha = 0.96)$$
At high temperatures, the algorithm explores broadly. As $T$ approaches 0, the acceptance probability for worse moves approaches 0, turning Simulated Annealing into greedy local exploitation.

### Q7. Why do both algorithms start from the exact same initial timetable?
**Answer:** Starting from the identical initial schedule (generated via `random.seed(42)`) ensures an unbiased, fair comparison of convergence speed, iteration count, and optimization quality.

### Q8. What happens when you edit a class in the GUI and click "Apply Changes"?
**Answer:** The program validates that the entered subject, professor, day, slot, and room belong to valid domains. It then updates that class's record, recalculates the exact penalty and violation counts, and updates the weekly grid in real time.

### Q9. What does the line graph in the GUI show?
**Answer:** It compares penalty score on the Y-axis against search iterations on the X-axis for both algorithms. It visually proves that Hill Climbing is monotonically greedy, while Simulated Annealing explores with early fluctuations before settling into an optimal solution.

### Q10. What is a "local optimum" in timetable scheduling?
**Answer:** A local optimum is a timetable state where every immediate neighbor move has an equal or worse penalty, yet the schedule still has unresolved conflicts. Standard Hill Climbing gets trapped in local optima because it refuses uphill moves, whereas Simulated Annealing can escape using its probabilistic acceptance criterion.
