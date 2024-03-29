# ---
# title : Example Case 3 - Year Planning
# author : Andrew Rosemberg
# date : 15th Feb 2019
# ---

# # Introduction

# This an example of the HydroPowerModels package for solving a simple stochastic case with the following specifications:
#    - 3 Buses
#    - 3 Lines
#    - 2 Generators
#    - 1 Reservoir and Hydrogenerator
#    - 3 Scenarios
#    - 12 Stages

# # Case

# ## Importing package and optimizer
using MosekTools # Gurobi
using HydroPowerModels
using JuMP
using Statistics

# ## Initialization
using Random
seed = 1221

# ## Load Case Specifications

# Data
case = "bolivia" # bolivia, case3
formulation = SOCWRConicPowerModel # SOCWRConicPowerModel, DCPPowerModel
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
num_stages = 96 # 96, 60
rm_stages = 0 # 0, 12

# Parameters
params = create_param(;
    stages = num_stages,
    model_constructor_grid = formulation,
    post_method = PowerModels.build_opf,
    optimizer = Mosek.Optimizer,
    # discount_factor=0.99502487562
);

# ## Build Model
m = hydro_thermal_operation(alldata, params);

# # ## Save subproblem
# results = HydroPowerModels.simulate(m, 2);
# model = m.forward_graph[1].subproblem
# delete(model, all_variables(model)[findfirst(x -> x == "",  name.(all_variables(model)))])
# JuMP.write_to_file(model, joinpath(case_dir, string(formulation)) * ".mof.json")

# ## Train
Random.seed!(seed)
start_time = time()
HydroPowerModels.train(
    m;
    iteration_limit = 2000,
    stopping_rules = [SDDP.Statistical(; num_replications = 300, iteration_period = 500)],
);
end_time = time() - start_time

# Termination Status and solve time (s)
(SDDP.termination_status(m.forward_graph), end_time)

# ## Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 300);

# ## Objective
objective_values = [sum(results[:simulations][i][t][:stage_objective] for t=1:num_stages-rm_stages) for i=1:length(results[:simulations])]
mean(objective_values)