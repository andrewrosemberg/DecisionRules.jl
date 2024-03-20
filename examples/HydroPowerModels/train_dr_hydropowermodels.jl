using Statistics
using Random
using Flux
using DecisionRules
using Gurobi
import ParametricOptInterface as POI
using Wandb, Dates, Logging

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

identity(x) = x

# Parameters

case_name = "case3"
formulation = "DCPPowerModel"
formulation_file = formulation * ".mof.json"
num_stages = 48
num_train_samples = 10000
# num_samples = 100
dense = Dense # RNN, Dense
activation = identity # tanh, identity
layers = Int64[] # [8, 8], Int64[]
num_models = num_stages # 1, num_stages
ensure_feasibility = ensure_feasibility_sigmoid
optimizer=Flux.Adam(0.01)

# Build MSP

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)
num_hydro = length(initial_state)
for subproblem in subproblems
    set_optimizer(subproblem, () -> POI.Optimizer(Gurobi.Optimizer()))
end

# Logging

lg = WandbLogger(
    project = "HydroPowerModels",
    name = "$(case_name)-$(formulation)-h$(num_stages)-$(now())",
    config = Dict(
        "layers" => layers,
        "activation" => string(activation),
        "num_models" => num_models,
        "dense" => string(dense),
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizer)
    )
)

function record_loss(iter, loss)
    Wandb.log(lg, Dict("metrics/loss" => loss))
    return nothing
end

# Build Model
models = dense_multilayer_nn(num_models, num_hydro, num_hydro, layers; activation=activation, dense=dense)

# Random.seed!(222)
# objective_values = [simulate_multistage(
#     subproblems, state_params_in, state_params_out, 
#     initial_state, sample(uncertainty_samples), 
#     models;
#     ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
# ) for _ in 1:num_samples]
# mean(objective_values)

# Train Model
train_multistage(models, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; 
    num_train_samples=num_train_samples, optimizer=optimizer,
    record_loss=record_loss,
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
)

# Finish the run
close(lg)

Random.seed!(222)
objective_values = [simulate_multistage(
    subproblems, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    models;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
) for _ in 1:100]
mean(objective_values)

# ################# Test MultiNetwork ################

# models = [Dense(num_hydro * (i +1) => num_hydro) for i in 1:num_stages]

# Random.seed!(222)
# objective_values = [simulate_multistage(
#     subproblems, state_params_in, state_params_out, 
#     initial_state, sample(uncertainty_samples), 
#     models;
#     ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
# ) for _ in 1:num_samples]
# mean(objective_values)

# train_multistage(models, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; num_train_samples=num_train_samples,
#     ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
# )

# Random.seed!(222)
# objective_values = [simulate_multistage(
#     subproblems, state_params_in, state_params_out, 
#     initial_state, sample(uncertainty_samples), 
#     models;
#     ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
# ) for _ in 1:num_samples]
# mean(objective_values)