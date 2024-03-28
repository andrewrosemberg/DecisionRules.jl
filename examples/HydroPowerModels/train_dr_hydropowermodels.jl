using Statistics
using Random
using Flux
using DecisionRules
using Gurobi
import ParametricOptInterface as POI
using Wandb, Dates, Logging

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions
identity(x) = x

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "case3"
formulation = "DCPPowerModel"
num_stages = 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
save_file = "$(case_name)-$(formulation)-h$(num_stages)-$(now())"
formulation_file = formulation * ".mof.json"
num_epochs=5
num_batches=5000
num_train_per_batch=1
dense = Dense # RNN, Dense
activation = identity # tanh, identity
layers = Int64[] # Int64[8, 8], Int64[]
num_models = num_stages # 1, num_stages
ensure_feasibility = non_ensurance # ensure_feasibility_double_softplus
optimizer=Flux.Adam(0.01)

# Build MSP

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)
set_optimizer(det_equivalent, () -> POI.Optimizer(Gurobi.Optimizer()))
set_attributes(det_equivalent, "OutputFlag" => 0)

num_hydro = length(initial_state)
# for subproblem in subproblems
#     set_optimizer(subproblem, () -> POI.Optimizer(Gurobi.Optimizer()))
#     set_attributes(subproblem, "OutputFlag" => 0)
# end

# Logging

lg = WandbLogger(
    project = "HydroPowerModels",
    name = save_file,
    config = Dict(
        "layers" => layers,
        "activation" => string(activation),
        "num_models" => num_models,
        "dense" => string(dense),
        "ensure_feasibility" => string(ensure_feasibility),
        "optimizer" => string(optimizer)
    )
)

function record_loss(iter, model, loss, tag)
    Wandb.log(lg, Dict(tag => loss))
    return false
end

# Build Model
models = dense_multilayer_nn(num_models, num_hydro, num_hydro, layers; activation=activation, dense=dense)

Random.seed!(222)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    initial_state, sample(uncertainty_samples), 
    models;
    ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume)
) for _ in 1:100]
best_obj = mean(objective_values)

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(best_obj, model_path, 0.003)

adjust_hyperparameters = (iter, opt_state, num_train_per_batch) -> begin
    if iter % 1100 == 0
        num_train_per_batch = num_train_per_batch * 2
    end
    return num_train_per_batch
end

# Train Model
for iter in 1:num_epochs
    train_multistage(models, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples; 
        num_batches=num_batches,
        num_train_per_batch=num_train_per_batch,
        optimizer=optimizer,
        record_loss= (iter, model, loss, tag) -> begin
            if tag == "metrics/training_loss"
                save_control(iter, model, loss)
            end
            return record_loss(iter, model, loss, tag)
        end,
        # ensure_feasibility=(x_out, x_in, uncertainty) -> ensure_feasibility(x_out, x_in, uncertainty, max_volume),
        adjust_hyperparameters=adjust_hyperparameters
    )
end

# Finish the run
close(lg)
