module LeximinRotaLineAssignment

using JuMP, Random
import HiGHS

export assign_residents_to_lines

function assign_residents_to_lines(preferences, rng = Random.default_rng())
    num_residents, num_lines = size(preferences)
    rng = rng isa Integer ? Random.Xoshiro(rng) : rng

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, assigned[1:num_residents, 1:num_lines], Bin)

    @constraint(model, [resident in 1:num_residents], 
        sum(assigned[resident, line] for line in 1:num_lines) == 1
    )
    @constraint(model, [line in 1:num_lines], 
        div(num_residents, num_lines) <= sum(assigned[resident, line] for resident in 1:num_residents) <= cld(num_residents, num_lines)
    )

    for rank in num_lines:-1:2
        count_at_rank = sum(assigned[resident, preferences[resident, rank]] for resident in 1:num_residents)
        @objective(model, Min, count_at_rank)
        optimize!(model)
        @constraint(model, count_at_rank <= round(Int, objective_value(model)))
    end

    @objective(model, Min, sum(rand(rng) * assigned[resident, line] for resident in 1:num_residents, line in 1:num_lines))
    optimize!(model)

    return [argmax(line -> value(assigned[resident, line]), 1:num_lines) for resident in 1:num_residents]
end

end # module
