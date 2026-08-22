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

# Mathematical Guarantees
1. **Leximin Optimality (Lorenz Dominance):** Sequentially minimizes the number of residents 
   assigned to their \$M\$-th choice, \$(M-1)\$-th choice, ..., down to 2nd choice.
2. **Equitable Balancing:** Line counts are strictly bounded within `⌊N/M⌋` and `⌈N/M⌉`.
3. **Unbiased Tie-Breaking:** If multiple assignments share the exact same leximin profile,
   ties are broken via i.i.d. random linear perturbations across valid extreme points.

# Arguments
- `preferences::AbstractMatrix{<:Integer}`: An \$N \\times M\$ matrix where row \$r\$ represents 
  resident \$r\$'s ranked preference of rota lines from 1st choice (col 1) to \$M\$-th choice (col \$M\$). 
  Each row must be a valid permutation of `1:M`.
- `seed::Union{Integer, Nothing}`: (Optional) Seed for deterministic tie-breaking. 
  Defaults to `nothing` (uses thread-safe system entropy).

# Returns
- `Vector{Int}`: Assigned rota line ID (1 to \$M\$) for each resident doctor (1 to \$N\$).
"""
function assign_residents_to_lines(
    preferences::AbstractMatrix{<:Integer}; 
    seed::Union{Integer, Nothing} = nothing
)::Vector{Int}
    # 1. 1-based indexing and dimension validation
    Base.require_one_based_indexing(preferences)
    num_residents, num_lines = size(preferences)

    if num_residents == 0 || num_lines == 0
        throw(ArgumentError("Preferences matrix must be non-empty."))
    end

    # 2. Zero-allocation row permutation validation
    for r in 1:num_residents
        if !isperm(@view preferences[r, :])
            throw(ArgumentError("Row $r must be a valid permutation of line IDs 1:$num_lines."))
        end
    end

    # 3. Trivial fast-paths (zero solver overhead)
    if num_lines == 1
        return fill(1, num_residents)
    end
    if num_residents == 1
        return [preferences[1, 1]]
    end

    # 4. Balanced line capacity bounds: ⌊N/M⌋ and ⌈N/M⌉
    min_capacity = div(num_residents, num_lines)
    max_capacity = cld(num_residents, num_lines)

    # 5. Model setup with HiGHS & MultiObjectiveAlgorithms
    # Sub-solver attributes are set directly on the inner HiGHS instance
    model = Model() do
        sub_optimizer = HiGHS.Optimizer()
        MOI.set(sub_optimizer, MOI.RawOptimizerAttribute("mip_feasibility_tolerance"), 1e-8)
        MOI.set(sub_optimizer, MOI.RawOptimizerAttribute("primal_feasibility_tolerance"), 1e-8)
        return MOA.Optimizer(sub_optimizer)
    end

    set_silent(model)
    set_attribute(model, MOA.Algorithm(), MOA.Lexicographic())
    set_attribute(model, MOA.LexicographicAllPermutations(), false)

    # 6. Decision variables & constraints
    @variable(model, assignment[1:num_residents, 1:num_lines], Bin)

    # Constraint: Each resident doctor gets exactly one rota line
    @constraint(model, [r in 1:num_residents], 
        sum(assignment[r, l] for l in 1:num_lines) == 1
    )

    # Constraint: Balanced rota line capacities
    @constraint(model, [l in 1:num_lines], 
        min_capacity <= sum(assignment[r, l] for r in 1:num_residents) <= max_capacity
    )

    # 7. Leximin Objectives: sequentially minimize worst choice (rank M) down to 2nd choice
    # Uses JuMP macro expansion to build MutableAffineExpressions without intermediate array allocation
    objectives = [
        @expression(model, sum(assignment[r, preferences[r, k]] for r in 1:num_residents))
        for k in num_lines:-1:2
    ]

    # Final tie-breaker: Thread-safe, non-allocating objective builder
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    noise = rand(rng, num_residents, num_lines)
    
    tie_breaker = @expression(
        model, 
        sum(noise[r, l] * assignment[r, l] for r in 1:num_residents, l in 1:num_lines)
    )
    push!(objectives, tie_breaker)

    @objective(model, Min, objectives)

    # 8. Solve and verify feasibility
    optimize!(model)
    if !is_solved_and_feasible(model; allow_suboptimal = false)
        error("Optimization failed. Solver status: $(termination_status(model)), Primal status: $(primal_status(model))")
    end

    # 9. Robust threshold extraction
    result = Vector{Int}(undef, num_residents)
    for r in 1:num_residents
        assigned_line = findfirst(l -> value(assignment[r, l]) > 0.5, 1:num_lines)
        if assigned_line === nothing
            error("Numerical anomaly: No active binary assignment found for resident $r.")
        end
        result[r] = assigned_line
    end

    return result
end

end # module LeximinRotaLineAssignment
