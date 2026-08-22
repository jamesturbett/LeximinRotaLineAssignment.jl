# LeximinRotaLineAssignment.jl

LICENSE: PolyForm Noncommercial 1.0.0

Resident doctors rank all starting lines in the generic work schedule in descending order of preference. A leximin-optimal assignment is found, with ties broken between equally optimal solutions through the introduction of random noise.

- **HiGHS** solver
- **JuMP (Julia)** modelling language for mathematical optimisation

## 🚀 Quick Start

### Installation

Add the required packages to your Julia environment:

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Random"])
```

---

### Basic Example

Suppose you have **6 resident doctors** choosing between **3 rota lines** (Line 1, Line 2, Line 3). Line capacities will automatically balance to `6 ÷ 3 = 2` residents per line.

Each row in the preference matrix represents a resident's ranked choices from **1st choice (col 1)** to **last choice (col 3)**:

```julia
using LeximinRotaLineAssignment

# Preferences: 6 residents (rows) ranking 3 lines (columns 1st -> 3rd choice)
preferences = [
    1  2  3;  # Resident 1: prefers Line 1 > Line 2 > Line 3
    1  3  2;  # Resident 2: prefers Line 1 > Line 3 > Line 2
    2  1  3;  # Resident 3: prefers Line 2 > Line 1 > Line 3
    2  3  1;  # Resident 4: prefers Line 2 > Line 3 > Line 1
    3  1  2;  # Resident 5: prefers Line 3 > Line 1 > Line 2
    3  2  1;  # Resident 6: prefers Line 3 > Line 2 > Line 1
]

# Compute the leximin-optimal assignment
assignments = assign_residents_to_lines(preferences)

println("Assigned Lines: ", assignments)
# Output: Assigned Lines: [1, 1, 2, 2, 3, 3]
```

---

### Deterministic / Seeded Runs

Pass an integer seed (or any `Random.AbstractRNG`) to guarantee reproducible tie-breaking:

```julia
# Deterministic tie-breaking using an integer seed
assignments = assign_residents_to_lines(preferences, 42)

# Or pass a custom random number generator
using Random
rng = Xoshiro(1234)
assignments = assign_residents_to_lines(preferences, rng)
```

---

### Verifying Assignment Quality

You can easily inspect which preference rank each resident received:

```julia
for resident in 1:size(preferences, 1)
    assigned_line = assignments[resident]
    rank_achieved = findfirst(==(assigned_line), preferences[resident, :])
    println("Resident $resident -> Line $assigned_line (Choice #$rank_achieved)")
end
```

**Output:**
```text
Resident 1 -> Line 1 (Choice #1)
Resident 2 -> Line 1 (Choice #1)
Resident 3 -> Line 2 (Choice #1)
Resident 4 -> Line 2 (Choice #1)
Resident 5 -> Line 3 (Choice #1)
Resident 6 -> Line 3 (Choice #1)
```

---

### How Line Balancing Works
* If **$N$ is divisible by $M$** (e.g., 6 residents, 3 lines), every line gets exactly $N/M = 2$ residents.
* If **$N$ is not divisible by $M$** (e.g., 8 residents, 3 lines), lines are equitably bounded between $\lfloor 8/3 \rfloor = 2$ and $\lceil 8/3 \rceil = 3$ residents (capacities will be `3, 3, 2`).
