using Statistics
using Random
using Flux
using DecisionRules
using Gurobi
import ParametricOptInterface as POI

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

identity(x) = x

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, "case3")
)
for subproblem in subproblems
    set_optimizer(subproblem, () -> POI.Optimizer(Gurobi.Optimizer()))
end

num_samples = 10
Random.seed!(222)
model = RNN(1 => 1, identity)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), [model for _ in 1:length(subproblems)]
) for _ in 1:num_samples]
mean(objective_values)

for epoch in 1:3
    train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=100)
    Random.seed!(222)
    objective_values_after_train = [simulate_multistage(
        subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), [model for _ in 1:length(subproblems)]
    ) for _ in 1:num_samples]
    println(mean(objective_values_after_train))
end