# LeximinRotaLineAssignment.jl

LICENSE: PolyForm Noncommercial 1.0.0

Resident doctors rank all starting lines in the generic work schedule in descending order of preference. A leximin-optimal assignment is found, with ties broken between equally optimal solutions through the introduction of random noise.

- **HiGHS** solver
- **JuMP (Julia)** modelling language for mathematical optimisation

## Quick Start

```julia
using LeximinRotaLineAssignment

# Preference matrix: Rows = Doctors (1 to 4), Columns = Ranks (1st choice to 4th choice)
# Values represent Rota Line IDs (1 to 4)
preferences = [
    1  2  3  4;   # Doctor 1 prefers Line 1 > Line 2 > Line 3 > Line 4
    1  3  2  4;   # Doctor 2 prefers Line 1 > Line 3 > Line 2 > Line 4
    2  1  4  3;   # Doctor 3 prefers Line 2 > Line 1 > Line 4 > Line 3
    4  1  2  3    # Doctor 4 prefers Line 4 > Line 1 > Line 2 > Line 3
]

# Compute the leximin-optimal assignment (with optional deterministic seed)
assignments = assign_residents_to_lines(preferences; seed = 42)

println("Assigned Rota Lines:")
for (doctor, line) in enumerate(assignments)
    println("Doctor \$doctor -> Line \$line")
end
