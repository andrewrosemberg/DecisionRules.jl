# ## Importing package and optimizer
using Ipopt # Gurobi, MosekTools, Ipopt
using HydroPowerModels
using JuMP
using Statistics
import SDDP

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
formulation = ACPPowerModel # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
formulation_b = SOCWRConicPowerModel

# Parameters
params = create_param(;
    stages = num_stages,
    model_constructor_grid = formulation,
    post_method = PowerModels.build_opf,
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6, "mu_init"=>1e-4, "max_iter" => 6000),
    # discount_factor=0.99502487562
);

# ## Build Model
m = hydro_thermal_operation(alldata, params);

# ## Load Policy
SDDP.read_cuts_from_file(m.forward_graph,joinpath(case_dir, string(formulation), string(formulation_b)*"-"*string(formulation)*".cuts.json"))

# ## Simulation
using Random: Random
Random.seed!(seed)
results = HydroPowerModels.simulate(m, 10);

# ## Objective
objective_values = [sum(results[:simulations][i][t][:stage_objective] for t=1:num_stages-rm_stages) for i=1:length(results[:simulations])]
println("Mean Sim: ", mean(objective_values))
