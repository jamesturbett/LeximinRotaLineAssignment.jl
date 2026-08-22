module LeximinRotaLineAssignment

using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA
using Random

export assign_residents_to_lines

"""
    assign_residents_to_lines(
        preferences::AbstractMatrix{<:Integer}; 
        seed::Union{Integer, Nothing} = nothing
    ) -> Vector{Int}

Computes a provably fair, leximin-optimal assignment of resident doctors to rota lines 
on a Generic Work Schedule (GWS) based on submitted ordinal preference rankings.

# Arguments
- `preferences::AbstractMatrix{<:Integer}`: An \$N \\times M\$ matrix where row \$r\$ represents 
  resident \$r\$'s ranked preference of rota lines from 1st choice (col 1) to \$M\$-th choice (col \$M\$). 
  Each row must be a valid permutation of `1:M`.
- `seed::Union{Integer, Nothing}`: (Optional) Seed for deterministic tie-breaking. 
  Defaults to `nothing` (uses fresh system entropy).

# Returns
- `Vector{Int}`: Assigned rota line ID (1 to \$M\$) for each resident doctor (1 to \$N\$).
"""
function assign_residents_to_lines(
    preferences::AbstractMatrix{<:Integer}; 
    seed::Union{Integer, Nothing} = nothing
)::Vector{Int}
    num_residents, num_lines = size(preferences)

    # 1. Validation & edge cases
    (num_residents > 0 && num_lines > 0 && axes(preferences) == (1:num_residents, 1:num_lines)) || 
        throw(ArgumentError("Preferences must be a non-empty, standard 1-indexed (N × M) matrix."))

    for r in 1:num_residents
        if sort(@view preferences[r, :]) != 1:num_lines
            throw(ArgumentError("Row $r must be a valid permutation of lines 1:$num_lines."))
        end
    end

    if num_lines == 1
        return fill(1, num_residents)
    end

    # 2. Balanced line capacity bounds: ⌊N/M⌋ and ⌈N/M⌉
    min_capacity = div(num_residents, num_lines)
    max_capacity = cld(num_residents, num_lines)

    # 3. Model setup with HiGHS & MultiObjectiveAlgorithms
    model = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    set_attribute(model, MOA.Algorithm(), MOA.Lexicographic())
    set_attribute(model, MOA.LexicographicAllPermutations(), false)

    # 4. Decision variables & constraints
    @variable(model, assignment[1:num_residents, 1:num_lines], Bin)

    # Each resident doctor gets exactly one starting line
    @constraint(model, [r in 1:num_residents], 
        sum(assignment[r, l] for l in 1:num_lines) == 1
    )
    # Balanced rota line capacity (interval constraint)
    @constraint(model, [l in 1:num_lines], 
        min_capacity <= sum(assignment[r, l] for r in 1:num_residents) <= max_capacity
    )

    # 5. Leximin Objectives: sequentially minimize worst choice (rank M) down to 2nd choice
    objectives = [
        sum(assignment[r, preferences[r, k]] for r in 1:num_residents)
        for k in num_lines:-1:2
    ]

    # Final tie-breaker objective (O(1) push without array splatting)
    rng = isnothing(seed) ? Xoshiro() : Xoshiro(seed)
    noise = rand(rng, num_residents, num_lines)
    push!(objectives, sum(noise[r, l] * assignment[r, l] for r in 1:num_residents, l in 1:num_lines))

    @objective(model, Min, objectives)

    # 6. Solve and verify feasibility
    optimize!(model)
    if !is_solved_and_feasible(model; allow_suboptimal = false)
        error("Optimization failed. Solver status: $(termination_status(model)), Primal status: $(primal_status(model))")
    end

    # 7. Return assigned line IDs
    return [argmax(l -> value(assignment[r, l]), 1:num_lines) for r in 1:num_residents]
end

end # module LeximinRotaLineAssignment
