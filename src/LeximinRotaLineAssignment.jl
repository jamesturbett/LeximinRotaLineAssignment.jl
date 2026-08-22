module LeximinRotaLineAssignment

using JuMP, HiGHS, Random

export assign_residents_to_lines

function assign_residents_to_lines(preferences; seed = nothing)
    N, M = size(preferences)
    (N == 0 || M == 0) && return Int[]

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

    # Unbiased tie-breaker over optimal leximin face
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    @objective(model, Min, sum(rand(rng) * x[r, l] for r in 1:N, l in 1:M))
    optimize!(model)

    return [argmax(l -> value(x[r, l]), 1:M) for r in 1:N]
end

end # module
