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
using HiGHS
using HydroPowerModels
using JuMP
using Statistics

# ## Initialization
using Random
seed = 1221

# ## Load Case Specifications

# Data
case = "bolivia"
formulation = DCPPowerModel
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);

# Parameters
params = create_param(;
    stages = 96,
    model_constructor_grid = DCPPowerModel,
    post_method = PowerModels.build_opf,
    optimizer = HiGHS.Optimizer,
);

# ## Build Model
m = hydro_thermal_operation(alldata, params);

# ## Train
Random.seed!(seed)
start_time = time()
HydroPowerModels.train(
    m;
    iteration_limit = 100,
    stopping_rules = [SDDP.Statistical(; num_replications = 20, iteration_period = 20)],
);
end_time = time() - start_time

# Termination Status and solve time (s)
(SDDP.termination_status(m.forward_graph), end_time)

# ## Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 100);
results

# ## Objective
objective_values = [results[:simulations][i][1][:objective] for i=1:length(results[:simulations])]
mean(objective_values)

# ## Save subproblem
JuMP.write_to_file(m.forward_graph[1].subproblem, joinpath(case_dir, string(formulation)) * ".mof.json")
