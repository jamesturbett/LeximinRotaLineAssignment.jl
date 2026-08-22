module LeximinRotaLineAssignment

using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA
using Random

export assign_residents_to_lines

"""
    assign_residents_to_lines(preferences::AbstractMatrix{<:Integer}; seed = nothing)

Computes a provably fair, leximin-optimal assignment of resident doctors to rota lines 
on a Generic Work Schedule (GWS) based on submitted preference rankings.
"""
function assign_residents_to_lines(
    preferences::AbstractMatrix{<:Integer}; 
    seed::Union{Integer, Nothing} = nothing
)
    num_residents, num_lines = size(preferences)

    # 1. Validation & edge cases
    num_residents > 0 && num_lines > 0 || throw(ArgumentError("Preferences matrix cannot be empty."))
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

    # 3. Model setup
    model = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
    set_silent(model)
    set_attribute(model, MOA.Algorithm(), MOA.Lexicographic())
    set_attribute(model, MOA.LexicographicAllPermutations(), false)

    # 4. Decision variables & constraints
    @variable(model, assignment[1:num_residents, 1:num_lines], Bin)

    # Each resident gets exactly one starting line
    @constraint(model, [r in 1:num_residents], 
        sum(assignment[r, l] for l in 1:num_lines) == 1
    )
    # Balanced line capacity (interval constraint)
    @constraint(model, [l in 1:num_lines], 
        min_capacity <= sum(assignment[r, l] for r in 1:num_residents) <= max_capacity
    )

    # 5. Leximin Objectives (worst choice down to 2nd choice + tie-breaker)
    rank_counts = [
        sum(assignment[r, preferences[r, k]] for r in 1:num_residents)
        for k in num_lines:-1:2
    ]

    rng = isnothing(seed) ? Xoshiro() : Xoshiro(seed)
    noise = rand(rng, num_residents, num_lines)
    tie_breaker = sum(noise[r, l] * assignment[r, l] for r in 1:num_residents, l in 1:num_lines)

    @objective(model, Min, [rank_counts..., tie_breaker])

    # 6. Solve and verify
    optimize!(model)
    if !is_solved_and_feasible(model; allow_suboptimal = false)
        error("Optimization failed with solver status: $(termination_status(model))")
    end

    # 7. Return assigned line IDs
    return [argmax(l -> value(assignment[r, l]), 1:num_lines) for r in 1:num_residents]
end

end # module LeximinRotaLineAssignment
