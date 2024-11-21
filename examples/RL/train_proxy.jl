using DecisionRules
using Flux

include("../HydroPowerModels/load_hydropowermodels.jl")
include("proxy_utils.jl")

case_name = "bolivia" # bolivia, case3
formulation = "DCPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 96 # 96, 48

(
    subproblems,
    state_params_in,
    state_params_out,
    uncertainty_samples,
    initial_state,
    max_volume
) = build_hydropowermodels(
    joinpath("../HydroPowerModels", case_name),
    formulation * ".mof.json";
    num_stages=num_stages,
    param_type=:Var
)

_std = make_standard_form_data(subproblems[1])
M, N = size(_std.A)

model = Chain(
    Dense(M, M, relu),
    Dense(M, M, relu),
    Dense(M, M, relu),
    Dense(M, M, relu),
)

optimizer = Flux.setup(Adam(), model)
E = 100

train_set = ???

for epoch in 1:E
    prev_std = nothing
    for data in train_set
        # subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out_target = data
        std = get_standard_form_data(data...)
        # TODO: we should be able to reuse _std.A/c/l/u, just need to figure out how to build b on demand

        # sanity check
        if !isnothing(prev_std)
            if (prev_std.A != std.A) 
                @warn "Standard form A has changed"
            # elseif (prev_std.b != std.b)
            #     @warn "Standard form b has changed"
            elseif (prev_std.c != std.c)
                @warn "Standard form c has changed"
            elseif (prev_std.l != std.l)
                @warn "Standard form l has changed"
            elseif (prev_std.u != std.u)
                @warn "Standard form u has changed"
            end
        end


        val, grads = Flux.withgradient(model) do m
            y = m(std.b)
            dual_objective(y, std.A, std.b, std.c, std.l, std.u)
        end
        Flux.update!(optimizer, model, grads[1])

        prev_std = std
    end
end
