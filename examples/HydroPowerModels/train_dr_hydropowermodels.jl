using Statistics
using Random
using Flux
using DecisionRules
using Ipopt # Gurobi, MosekTools, Ipopt, MadNLP
# import CUDA # if error run CUDA.set_runtime_version!(v"12.1.0")
# CUDA.set_runtime_version!(v"12.1.0")
# using MadNLP 
# using MadNLPGPU
import ParametricOptInterface as POI
using Wandb, Dates, Logging
using JLD2

HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions
identity(x) = x

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "case3" # bolivia, case3
formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 48 # 96, 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
mkpath(model_dir)
save_file = "$(case_name)-$(formulation)-h$(num_stages)-$(now())"
formulation_file = formulation * ".mof.json"
num_epochs=2
num_batches=1000
_num_train_per_batch=3
dense = Dense # RNN, Dense
activation = relu # tanh, identity
layers = Int64[8, 8] # Int64[8, 8], Int64[]
num_models = num_stages # 1, num_stages
ensure_feasibility = non_ensurance # ensure_feasibility_double_softplus
optimizers= [Flux.Adam(0.01), Flux.Descent(0.1)] # Flux.Adam(0.01), Flux.Descent(0.1)
pre_trained_model = nothing # joinpath(HydroPowerModels_dir, case_name, "ACPPowerModel/models/case3-ACPPowerModel-h48-2024-04-15T18:07:30.145.jld2")

# Build MSP

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)

# ipopt = MadNLP.Optimizer(
#     linear_solver=LapackCPUSolver,
#     print_level=MadNLP.ERROR
# )
ipopt = Ipopt.Optimizer()
MOI.set(ipopt, MOI.RawOptimizerAttribute("print_level"), 0)
cached =
    () -> MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            ipopt,
        ),
        Float64,
)
POI_cached_optimizer() = POI.Optimizer(cached())

set_optimizer(det_equivalent, () -> POI_cached_optimizer())
# set_optimizer(det_equivalent, () -> Mosek.Optimizer())
# set_attribute(det_equivalent, "QUIET", true)
# set_attributes(det_equivalent, "OutputFlag" => 0)

num_hydro = length(initial_state)
# for subproblem in subproblems
#     set_optimizer(subproblem, () -> POI.Optimizer(Ipopt.Optimizer()))
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
        "optimizer" => string(optimizers)
    )
)

function record_loss(iter, model, loss, tag)
    Wandb.log(lg, Dict(tag => loss))
    return false
end

# Build Model
models = dense_multilayer_nn(num_models, num_hydro, num_hydro, layers; activation=activation, dense=dense)
if !isnothing(pre_trained_model)
    model = if num_models > 1
        DecisionRules.make_single_network(models, num_hydro)
    else
        models
    end
    model_save = JLD2.load(pre_trained_model)
    model_state = model_save["model_state"]
    Flux.loadmodel!(model, model_state)
end

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
    num_train_per_batch = _num_train_per_batch
    train_multistage(models, initial_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples; 
        num_batches=num_batches,
        num_train_per_batch=num_train_per_batch,
        optimizer=optimizers[floor(min(iter, length(optimizers)))],
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
