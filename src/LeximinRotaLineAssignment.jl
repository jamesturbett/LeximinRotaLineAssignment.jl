module LeximinRotaLineAssignment

using JuMP, HiGHS, Random

export assign_residents_to_lines

"""
    assign_residents_to_lines(preferences, [rng/seed]) -> Vector{Int}

Computes a balanced, leximin-optimal assignment of resident doctors to rota lines.

- `preferences`: \$N \\times M\$ matrix where row \$r\$ is a permutation of `1:M` (1st to \$M\$-th choice).
- `rng`: Optional `Random.AbstractRNG` or integer seed for reproducible tie-breaking.
"""
function assign_residents_to_lines(
    preferences::AbstractMatrix{<:Integer},
    rng::Random.AbstractRNG = Random.default_rng(),
)
    N, M = size(preferences)
    (N == 0 || M == 0) && return Int[]
    all(isperm, eachrow(preferences)) || throw(ArgumentError("Each row of preferences must be a permutation of 1:$M."))

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:N, 1:M], Bin)
    @constraint(model, [r in 1:N], sum(x[r, l] for l in 1:M) == 1)
    @constraint(model, [l in 1:M], div(N, M) <= sum(x[r, l] for r in 1:N) <= cld(N, M))

    # Sequentially minimize choice counts from worst (M) down to 2nd choice
    for k in M:-1:2
        count_k = sum(x[r, preferences[r, k]] for r in 1:N)
        @objective(model, Min, count_k)
        optimize!(model)
        @constraint(model, count_k <= round(Int, objective_value(model)))
    end

    # Unbiased tie-breaker over the optimal leximin face
    @objective(model, Min, sum(rand(rng) * x[r, l] for r in 1:N, l in 1:M))
    optimize!(model)

    if !is_solved_and_feasible(model)
        error("Solver failed to find an optimal assignment: $(termination_status(model))")
    end

    # Single batched C-call to retrieve solution matrix
    vals = value.(x)
    return [argmax(@view vals[r, :]) for r in 1:N]
end

# Multiple dispatch convenience method for integer seeds
assign_residents_to_lines(preferences, seed::Integer) =
    assign_residents_to_lines(preferences, Random.Xoshiro(seed))

end # module
