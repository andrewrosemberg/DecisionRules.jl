# ## Importing package and optimizer
using Gurobi # Gurobi, MosekTools
using Ipopt
using HydroPowerModels
using JuMP
using Statistics
import SDDP: stopping_rule_status, convergence_test, PolicyGraph, AbstractStoppingRule, Log
using Wandb, Dates, Logging

# ## Initialization
using Random
seed = 1221

# ## Load Case Specifications

# Data
case = "case3" # bolivia, case3
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
num_stages = 48 # 96, 48
rm_stages = 0 # 0, 12
formulation_b = DCPLLPowerModel # SOCWRConicPowerModel, DCPPowerModel, DCPLLPowerModel
formulation = ACPPowerModel

# Parameters
params = create_param(;
    stages = num_stages,
    model_constructor_grid = formulation_b,
    model_constructor_grid_forward = formulation,
    post_method = PowerModels.build_opf,
    optimizer = Gurobi.Optimizer,
    optimizer_forward = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "mu_init"=>1e-4, "max_iter" => 6000),
    # discount_factor=0.99502487562
);

# ## Build Model
m = hydro_thermal_operation(alldata, params);

# # ## Save subproblem
# results = HydroPowerModels.simulate(m, 2);
# model = m.forward_graph[1].subproblem
# delete(model, all_variables(model)[findfirst(x -> x == "",  name.(all_variables(model)))])
# JuMP.write_to_file(model, joinpath(case_dir, string(formulation)) * ".mof.json")

# Wandb logger

"""
    WandBLog(limit::Int)

Teriminate the algorithm after `limit` number of iterations.
"""
mutable struct WandBLog <: SDDP.AbstractStoppingRule
    lg
end

SDDP.stopping_rule_status(::WandBLog) = :not_solved

function SDDP.convergence_test(::SDDP.PolicyGraph, log::Vector{SDDP.Log}, rule::WandBLog)
    Wandb.log(rule.lg, Dict(
        "iteration" => length(log),
        "bound" => log[end].bound,
        "metrics/loss" => log[end].simulation_value,
    ))
    return false
end

save_file = "SDDP-$(case)-$(formulation)-h$(num_stages)-$(now())"

lg = WandbLogger(
    project = "HydroPowerModels",
    name = save_file,
)

# ## Train
Random.seed!(seed)
start_time = time()
HydroPowerModels.train(
    m;
    iteration_limit = 2000,
    stopping_rules = [WandBLog(lg); SDDP.Statistical(; num_replications = 300, iteration_period = 200)],
);
end_time = time() - start_time

# Termination Status and solve time (s)
(SDDP.termination_status(m.forward_graph), end_time)

# save cuts
SDDP.write_cuts_to_file(m.forward_graph,joinpath(case_dir, string(formulation), string(formulation_b)*"-"*string(formulation)*".cuts.json"))

# ## Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 300);

# ## Objective
objective_values = [sum(results[:simulations][i][t][:stage_objective] for t=1:num_stages-rm_stages) for i=1:length(results[:simulations])]
println("Mean Sim: ", mean(objective_values))

Wandb.log(rule.lg, Dict(
    "bound" => SDDP.calculate_bound(m.forward_graph),
    "metrics/final_loss" => mean(objective_values),
))

# Finish the run
close(lg)