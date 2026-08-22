module LeximinRotaLineAssignment

using JuMP, Random
import HiGHS

export assign_residents_to_lines

"""
    assign_residents_to_lines(preferences, [rng]) -> Vector{Int}

Computes a balanced, leximin-optimal assignment of resident doctors to rota lines.

# Arguments
- `preferences`: An \$N \\times M\$ matrix where row `r` lists line IDs (`1:M`) 
  in order of preference (column 1 is 1st choice, column \$M\$ is \$M\$-th choice).
- `rng`: Optional `Random.AbstractRNG` or integer seed for deterministic tie-breaking. 
  Defaults to `Random.default_rng()`.

# Mathematical Guarantees
1. **Leximin Fairness:** Sequentially minimizes the number of residents assigned to their
   worst choice (\$M\$-th), \$(M-1)\$-th, down to 2nd choice (which automatically maximizes 1st choices).
2. **Equitable Balancing:** Line loads are strictly bounded between `⌊N/M⌋` and `⌈N/M⌉`.
3. **Unbiased Tie-Breaking:** If multiple assignments share the exact same optimal leximin profile,
   ties are broken via random linear perturbation across extreme points of the optimal face.

# Returns
- `Vector{Int}`: Assigned rota line ID (`1:M`) for each resident (`1:N`).
"""
function assign_residents_to_lines(preferences, rng = Random.default_rng())
    num_residents, num_lines = size(preferences)

    # Normalize RNG: accepts either an integer seed or an AbstractRNG instance
    rng = rng isa Integer ? Random.Xoshiro(rng) : rng

    # Initialize the optimization model with the HiGHS solver in silent mode
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Binary decision variables: assigned[r, l] == 1 iff resident r is assigned to line l
    @variable(model, assigned[1:num_residents, 1:num_lines], Bin)

    # Constraint 1: Each resident must be assigned to exactly one rota line
    @constraint(model, [resident in 1:num_residents], 
        sum(assigned[resident, line] for line in 1:num_lines) == 1
    )

    # Constraint 2: Balanced capacities across all lines (bounded between ⌊N/M⌋ and ⌈N/M⌉)
    @constraint(model, [line in 1:num_lines], 
        div(num_residents, num_lines) <= sum(assigned[resident, line] for resident in 1:num_residents) <= cld(num_residents, num_lines)
    )

    # Leximin optimization stages:
    # Sequentially minimize choice counts from worst (num_lines) down to 2nd choice.
    # Minimizing choices M down to 2 automatically maximizes 1st choice assignments.
    for rank in num_lines:-1:2
        count_at_rank = sum(assigned[resident, preferences[resident, rank]] for resident in 1:num_residents)
        @objective(model, Min, count_at_rank)
        optimize!(model)
        
        # Lock the exact minimum count achieved as an upper bound for subsequent stages
        @constraint(model, count_at_rank <= round(Int, objective_value(model)))
    end

    # Tie-breaking stage:
    # Break any remaining ties over the optimal leximin face using random linear perturbation
    @objective(model, Min, sum(rand(rng) * assigned[resident, line] for resident in 1:num_residents, line in 1:num_lines))
    optimize!(model)

    # Extract the assigned line ID (1:num_lines) for each resident (1:num_residents)
    return [argmax(line -> value(assigned[resident, line]), 1:num_lines) for resident in 1:num_residents]
end

end # module
