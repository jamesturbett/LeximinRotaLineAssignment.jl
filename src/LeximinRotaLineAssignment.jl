module LeximinRotaLineAssignment

using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA
using Random

export assign_residents_to_lines, build_assignment_model

"""
    assign_residents_to_lines(
        preferences::AbstractMatrix{<:Integer}; 
        seed::Union{Integer, Random.AbstractRNG, Nothing} = nothing,
        optimizer = () -> MOA.Optimizer(HiGHS.Optimizer)
    ) -> Vector{Int}

High-level pipeline: Validates preference rankings, solves a leximin-optimal assignment 
model on a Generic Work Schedule (GWS), and returns the assigned rota line ID for each resident doctor.

# Arguments
- `preferences`: An (N × M) matrix of integers where each row represents a resident doctor 
  and columns are preference ranks (1st choice to M-th choice). Each row must be a valid permutation of `1:M`.
- `seed`: (Optional) Integer seed or `Random.AbstractRNG` for deterministic tie-breaking. 
  If `nothing` (default), a fresh random seed is drawn from system entropy via `Xoshiro()`.
- `optimizer`: JuMP-compatible optimizer constructor (defaults to MOA wrapping HiGHS).

# Returns
- `Vector{Int}`: Assigned rota line ID (1 to M) for each resident doctor (1 to N).
"""
function assign_residents_to_lines(
    preferences::AbstractMatrix{<:Integer}; 
    seed::Union{Integer, Random.AbstractRNG, Nothing} = nothing,
    optimizer = () -> MOA.Optimizer(HiGHS.Optimizer)
)::Vector{Int}
    num_residents, num_lines = size(preferences)

    # 1. Validate inputs
    _validate_preferences(preferences, num_residents, num_lines)

    # 2. Handle trivial single-line edge case
    if num_lines == 1
        return fill(1, num_residents)
    end

    # 3. Construct the multi-objective optimization model
    model, assignment = build_assignment_model(
        preferences; 
        seed = seed, 
        optimizer = optimizer
    )

    # 4. Solve the hierarchical leximin problem
    optimize!(model)

    # 5. Verify solver termination status and primal feasibility
    if !is_solved_and_feasible(model; allow_suboptimal = false)
        status = termination_status(model)
        p_status = primal_status(model)
        error("Optimization failed to find an optimal solution. Termination status: $status, Primal status: $p_status.")
    end

    # 6. Extract and return assigned line IDs
    return _extract_assignment(assignment, num_residents, num_lines)
end

"""
    build_assignment_model(
        preferences::AbstractMatrix{<:Integer}; 
        seed::Union{Integer, Random.AbstractRNG, Nothing} = nothing,
        optimizer = () -> MOA.Optimizer(HiGHS.Optimizer)
    ) -> Tuple{Model, Matrix{VariableRef}}

Constructs and returns the JuMP multi-objective leximin model along with the binary 
assignment decision variables without solving it. Useful for inspecting or adding custom constraints.
"""
function build_assignment_model(
    preferences::AbstractMatrix{<:Integer}; 
    seed::Union{Integer, Random.AbstractRNG, Nothing} = nothing,
    optimizer = () -> MOA.Optimizer(HiGHS.Optimizer)
)
    num_residents, num_lines = size(preferences)

    # Balanced rota line capacity bounds: ⌊N/M⌋ and ⌈N/M⌉
    min_capacity = div(num_residents, num_lines)
    max_capacity = cld(num_residents, num_lines)

    # Initialize model with MultiObjectiveAlgorithms (MOA) optimizer
    model = Model(optimizer)
    set_silent(model)
    
    # Configure Lexicographic solver attributes
    set_attribute(model, MOA.Algorithm(), MOA.Lexicographic())
    set_attribute(model, MOA.LexicographicAllPermutations(), false)
    
    # Strict tolerances guarantee integer rank counts do not degrade across priority levels
    set_attribute(model, MOA.AbsoluteTolerance(), 1e-5)
    set_attribute(model, MOA.RelativeTolerance(), 1e-5)

    # Binary decision variables: assignment[r, l] == 1 if resident r gets line l
    @variable(model, assignment[1:num_residents, 1:num_lines], Bin)

    # Constraint 1: Every resident doctor gets exactly one starting line
    @constraint(model, [r in 1:num_residents],
        sum(assignment[r, l] for l in 1:num_lines) == 1
    )

    # Constraint 2: Rota line upper capacity bound
    @constraint(model, [l in 1:num_lines],
        sum(assignment[r, l] for r in 1:num_residents) <= max_capacity
    )

    # Constraint 3: Rota line lower capacity bound (enforces balance when N >= M)
    if min_capacity > 0
        @constraint(model, [l in 1:num_lines],
            sum(assignment[r, l] for r in 1:num_residents) >= min_capacity
        )
    end

    # Hierarchical Leximin Objectives:
    # Sequentially minimize assignments to worst rank (M), then (M-1), down to rank 2.
    rank_counts = [
        sum(assignment[r, preferences[r, k]] for r in 1:num_residents)
        for k in num_lines:-1:2
    ]

    # Final-stage deterministic tie-breaker
    rng = _resolve_rng(seed)
    noise = rand(rng, num_residents, num_lines)
    tie_breaker = sum(noise[r, l] * assignment[r, l] for r in 1:num_residents, l in 1:num_lines)

    @objective(model, Min, [rank_counts..., tie_breaker])

    return model, assignment
end

# ----------------------------------------------------------------------
# Internal Helpers
# ----------------------------------------------------------------------

# Resolves seed or RNG instance safely without mutating global RNG state
_resolve_rng(rng::Random.AbstractRNG) = rng
_resolve_rng(seed::Integer)            = Xoshiro(seed)
_resolve_rng(::Nothing)                = Xoshiro()

# High-performance preference validation: O(N*M) time, O(M) auxiliary memory
function _validate_preferences(
    preferences::AbstractMatrix{<:Integer}, 
    num_residents::Int, 
    num_lines::Int
)
    num_residents > 0 || throw(ArgumentError("Preferences matrix must have at least 1 resident (rows > 0)."))
    num_lines > 0 || throw(ArgumentError("Preferences matrix must have at least 1 rota line (columns > 0)."))

    seen = falses(num_lines)
    for r in 1:num_residents
        fill!(seen, false)
        for k in 1:num_lines
            line = preferences[r, k]
            if line < 1 || line > num_lines
                throw(ArgumentError("Invalid line ID $line for resident $r at rank $k. Must be in 1:$num_lines."))
            end
            if seen[line]
                throw(ArgumentError("Duplicate line ID $line found in resident $r's preferences. Each row must be a permutation of 1:$num_lines."))
            end
            seen[line] = true
        end
    end
end

# Robust, type-stable extraction of binary assignments
function _extract_assignment(assignment, num_residents::Int, num_lines::Int)::Vector{Int}
    return [
        argmax(l -> value(assignment[r, l]), 1:num_lines)
        for r in 1:num_residents
    ]
end

end # module LeximinRotaLineAssignment
