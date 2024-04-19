using Statistics
using Random
using Flux
using DecisionRules
using Ipopt # Gurobi, MosekTools, Ipopt, MadNLP
import ParametricOptInterface as POI
using JLD2


HydroPowerModels_dir = dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

# Functions

function non_ensurance(x_out, x_in, uncertainty, max_volume)
    return x_out
end

# Parameters
case_name = "case3" # bolivia, case3
formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 48 # 96, 48
model_dir = joinpath(HydroPowerModels_dir, case_name, formulation, "models")
model_file = readdir(model_dir, join=true)[1] # edit this for a specific model
formulation_file = formulation * ".mof.json"
num_epochs=1
num_batches=2000
_num_train_per_batch=100
dense = Dense # RNN, Dense
activation = relu # tanh, DecisionRules.identity
layers = Int64[] # Int64[8, 8], Int64[]
num_models = num_stages # 1, num_stages
ensure_feasibility = non_ensurance # ensure_feasibility_double_softplus

# Build MSP

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages
)

det_equivalent, uncertainty_samples = DecisionRules.deterministic_equivalent(subproblems, state_params_in, state_params_out, initial_state, uncertainty_samples)

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

# Build Model
models = dense_multilayer_nn(num_models, num_hydro, num_hydro, layers; activation=activation, dense=dense)
model = if num_models > 1
    DecisionRules.make_single_network(models, num_hydro)
else
    models
end
model_save = JLD2.load(model_file)
model_state = model_save["model_state"]
Flux.loadmodel!(model, model_state)

Random.seed!(222)
num_samples = 100
objective_values = Vector{Float64}(undef, num_samples)
states = Vector{Any}(undef, num_samples)
inflows = Array{Float64,3}(undef, num_samples, num_hydro, num_stages)
record_variables_names = ["0_pg"]
record_variables = Dict{String, Any}()
record = Dict{String, Array{Float64,3}}()
for _var in record_variables_names
    num_vars = length(find_variables(det_equivalent, [_var; r"#1$"]))
    record[_var] = Array{Float64,3}(undef, num_samples, num_vars, num_stages)
    record_variables[_var] = [find_variables(det_equivalent, [_var; Regex("#$i\$")]) for i in 1:num_stages]
end
for i in 1:num_samples
    uncertainty_s = sample(uncertainty_samples)
    for j in 1:num_hydro, t in 1:num_stages
        inflow_var = collect(keys(uncertainty_s[t]))
        inflow_var = inflow_var[findfirst(x -> occursin("_inflow[$j]", name(x)), inflow_var)]
        inflows[i, j, t] = uncertainty_s[t][inflow_var]
    end
    objective_values[i] = simulate_multistage(
        det_equivalent, state_params_in, state_params_out, 
        initial_state, uncertainty_s, 
        models;
        ensure_feasibility=(x_out, x_in, _sa) -> ensure_feasibility(x_out, x_in, _sa, max_volume)
    )
    for _var in record_variables_names
        num_vars = length(find_variables(det_equivalent, [_var; r"#1$"]))
        for j in 1:num_vars
            record[_var][i, j, t] = value(record_variables[_var][t][j])
        end
    end
    states[i] = Vector{Vector{Float64}}(undef, num_hydro)
    for j in 1:num_hydro
        states[i][j] = Vector{Float64}(undef, num_stages+1)
        states[i][j][1] = initial_state[j]
        for t in 1:num_stages
            states[i][j][t+1] = value(state_params_out[t][j][2])
        end
    end
end

# Plot Volumes

using Plots
using Statistics

plt = plot(1:num_stages+1, [sum([states[1][j][t] for j in 1:num_hydro]) for t in 1:num_stages+1], legend=false, xlabel="Stage", ylabel="Volume (Hm3)", title="$(case_name)-$(formulation)");
for i in 2:num_samples
    plot!(plt, 1:num_stages+1, [sum([states[i][j][t] for j in 1:num_hydro]) for t in 1:num_stages+1]);
end
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, "$(case_name)-$(formulation)-Volume.png"))

# Plot Mean Volume

plt = plot(1:num_stages+1, [mean(sum([states[i][j][t] for j in 1:num_hydro]) for i in 1:num_samples) for t in 1:num_stages+1], xlabel="Stage", ylabel="Volume (Hm3)", label="Mean Volume", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, "$(case_name)-$(formulation)-MeanVolume.png"))

# Plot Mean Inflows

plt = plot(1:num_stages, [mean(sum(inflows[i, j, t] for j in 1:num_hydro) for i in 1:num_samples) for t in 1:num_stages], xlabel="Stage", ylabel="Inflow (Hm3)", label="Mean Inflow", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, "$(case_name)-$(formulation)-MeanInflow.png"))

# Plot Generation

plt = plot(1:num_stages, [mean(sum(record["0_pg"][i, j, t] for j in 1:num_hydro) for i in 1:num_samples) for t in 1:num_stages], xlabel="Stage", ylabel="Generation (MW)", label="Mean Generation", title="$(case_name)-$(formulation)");
savefig(plt, joinpath(HydroPowerModels_dir, case_name, formulation, "$(case_name)-$(formulation)-MeanGeneration.png"))
