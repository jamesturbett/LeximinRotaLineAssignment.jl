module LeximinRotaLineAssignment

using JuMP, Random
import HiGHS

export assign_residents_to_lines

"""
    assign_residents_to_lines(preferences, [rng]) -> Vector{Int}

Computes a balanced, leximin-optimal assignment of resident doctors to rota lines.

# Arguments
- `preferences`: An \$N \\times L\$ matrix where row \$r\$ lists rota lines (`1:L`) 
  in order of resident \$r\$'s preference (column 1 is 1st choice, column \$L\$ is \$L\$-th choice).
- `rng`: Optional `Random.AbstractRNG` or integer seed for deterministic tie-breaking. 
  Defaults to `Random.default_rng()`.

# Mathematical Guarantees
1. **Leximin Fairness:** Sequentially minimizes the number of residents assigned to their
   worst choice (\$L\$-th), \$(L-1)\$-th, down to 2nd choice (which automatically maximizes 1st choices).
2. **Equitable Balancing:** Line loads are strictly bounded between `⌊N/L⌋` and `⌈N/L⌉`.
3. **Unbiased Tie-Breaking:** If multiple assignments share the exact same optimal leximin profile,
   ties are broken via random linear perturbation across extreme points of the optimal face.

# Returns
- `Vector{Int}`: Assigned rota line ID (`1:L`) for each resident (`1:N`).
"""
function assign_residents_to_lines(preferences, rng = Random.default_rng())
    N, L = size(preferences)
    all(isperm, eachrow(preferences)) || throw(ArgumentError("Each row of preferences must be a permutation of line IDs 1:$L."))
    rng = rng isa Integer ? Random.Xoshiro(rng) : rng

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Binary decision variable: x[r, l] == 1 iff resident r is assigned to line l
    @variable(model, x[1:N, 1:L], Bin)

    # Each resident is assigned to exactly one rota line
    @constraint(model, [r in 1:N], sum(x[r, l] for l in 1:L) == 1)

    # Balanced capacities across all lines (bounded between ⌊N/L⌋ and ⌈N/L⌉)
    @constraint(model, [l in 1:L], div(N, L) <= sum(x[r, l] for r in 1:N) <= cld(N, L))

    # Leximin optimization: sequentially minimize resident counts at worst choice (L) down to 2nd choice
    for k in L:-1:2
        count_k = sum(x[r, preferences[r, k]] for r in 1:N)
        @objective(model, Min, count_k)
        optimize!(model)
        @constraint(model, count_k <= round(Int, objective_value(model)))
    end

    # Unbiased tie-breaker over the optimal leximin face
    @objective(model, Min, sum(rand(rng) * x[r, l] for r in 1:N, l in 1:L))
    optimize!(model)

    # Extract the assigned line ID (1:L) for each resident (1:N)
    return [argmax(l -> value(x[r, l]), 1:L) for r in 1:N]
end

end # module
