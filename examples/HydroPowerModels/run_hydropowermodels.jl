using Statistics
using Random
using Flux
using DecisionRules
using Gurobi
import ParametricOptInterface as POI

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

identity(x) = x

case_name = "case3"
formulation = "DCPPowerModel.mof.json"
num_stages=48

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation; num_stages=num_stages
)
num_hydro = length(initial_state)
for subproblem in subproblems
    set_optimizer(subproblem, () -> POI.Optimizer(Gurobi.Optimizer()))
end

num_samples = 10
Random.seed!(222)
model = RNN(num_hydro * 2 => num_hydro, identity)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    [model for _ in 1:length(subproblems)];
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)

num_epochs = 5

training_logs = Array{Float64}(undef, num_epochs)

for epoch in 1:num_epochs
    train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=500)
    Random.seed!(222)
    objective_values_after_train = [simulate_multistage(
        subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), [model for _ in 1:length(subproblems)];
        ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
    ) for _ in 1:num_samples]
    training_logs[epoch] = mean(objective_values_after_train)
end

objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), [model for _ in 1:length(subproblems)];
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)