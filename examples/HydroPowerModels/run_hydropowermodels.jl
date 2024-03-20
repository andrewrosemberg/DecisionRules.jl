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

num_samples = 30

############### Test RNN ################

# model = RNN(num_hydro => num_hydro, identity)
# model.state .= initial_state
model = Chain(RNN(num_hydro => num_hydro * 8), RNN(num_hydro * 8 => num_hydro * 8), RNN(num_hydro * 8 => num_hydro), RNN(num_hydro => num_hydro))

Random.seed!(222)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    model;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)

train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=1000,
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
)

Random.seed!(222)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, initial_state, sample(uncertainty_samples), model;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)

################# Test MultiNetwork ################

models = [Dense(num_hydro * (i +1) => num_hydro) for i in 1:num_stages]

Random.seed!(222)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    models;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)

train_multistage(models, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=100000,
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
)

Random.seed!(222)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    models;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility_sigmoid(x_out, x_in, uncertainty, max_volume)
) for _ in 1:num_samples]
mean(objective_values)