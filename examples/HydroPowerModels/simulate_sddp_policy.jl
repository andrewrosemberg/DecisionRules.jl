# ## Importing package and optimizer
# using Ipopt, HSL_jll # Gurobi, MosekTools, Ipopt
using MosekTools
using MadNLP
using HydroPowerModels
using JuMP
using Statistics
import SDDP

# ## Initialization
using Random
seed = 1221

# ## Load Case Specifications

# Data
case = "bolivia" # bolivia, case3
case_dir = joinpath(dirname(@__FILE__), case)
alldata = HydroPowerModels.parse_folder(case_dir);
for load in values(alldata[1]["powersystem"]["load"])
    load["qd"] = load["qd"] * 0.6
    load["pd"] = load["pd"] * 0.6
end
rm_stages = 30 # 30, 12
num_stages = 96 + rm_stages # 96, 48
formulation = DCPPowerModel # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
formulation_b = DCPPowerModel # DCPLLPowerModel

# Parameters
params = create_param(;
    stages = num_stages,
    model_constructor_grid = formulation,
    post_method = PowerModels.build_opf,
    # optimizer = Mosek.Optimizer,
    optimizer = Gurobi.Optimizer,
    # optimizer = ()->MadNLP.Optimizer(print_level=MadNLP.INFO)
    # optimizer = optimizer_with_attributes(Ipopt.Optimizer, 
    #     "print_level" => 0,
    #     "hsllib" => HSL_jll.libhsl_path,
    #     "linear_solver" => "ma27"
    # ),
    # discount_factor=0.99502487562
);

# ## Build Model
m = hydro_thermal_operation(alldata, params);

# ## Load Policy
SDDP.read_cuts_from_file(m.forward_graph,joinpath(case_dir, string(formulation), string(formulation_b)*"-"*string(formulation)*".cuts.json"))

# ## Simulation
using Random: Random
Random.seed!(seed)
num_sim = 10
results = HydroPowerModels.simulate(m, num_sim);

# Plot volume
nhyd = alldata[1]["hydro"]["nHyd"]
using Plots
using CSV
using DataFrames
volume_to_mw(volume, stage_hours; k = 0.0036) = volume / (k * stage_hours)

# labels = ["GML-DR"; "2St-LDR"; "DCLL"; "SOC"]
# colors = [:black :purple :red :blue]
# markers = [:hline :+ :pixel :xcross]

labels = ["GML-DR"; "2St-LDR"; "DCLL"]
colors = [:black :purple :red]
markers = [:hline :+ :pixel]

hydro_gen = [mean(sum(volume_to_mw(results[:simulations][i][t][:reservoirs][:reservoir][j].out, 1) for j=1:nhyd) for i=1:num_sim) for t=1:num_stages-rm_stages]

savefig(plot(hydro_gen, legend=false, xlabel="Stage", ylabel="Volume (Hm3)", title="$(case)-$(formulation_b)-$(formulation)")
, joinpath(case_dir, string(formulation), "SDDP-$(case)-$(formulation_b)-$(formulation)-Volume.png")
)
# 2.39577833453e-312
df = CSV.read(joinpath(case_dir, string(formulation), "MeanVolume.csv"), DataFrame; header=true)
df[!, "$(string(formulation_b))"] = hydro_gen
# df = DataFrame(sddp=thermal_gen)

CSV.write(joinpath(case_dir, string(formulation), "MeanVolume.csv"), df)

savefig(plot(Matrix(df[!,labels]), labels=permutedims(names(df[!,labels])), xlabel="Stage", ylabel="Volume (MWh)", color=colors, shape=markers)
, joinpath(case_dir, string(formulation), "Comparison-$(case)-Volume.png")
)

# generation
num_gen = length(results[:simulations][1][1][:powersystem]["solution"]["gen"])
hydro_idx = HydroPowerModels.idx_hydro(results[:data][1])
thermal_gen = [mean(sum(results[:simulations][i][t][:powersystem]["solution"]["gen"]["$j"]["pg"] * results[:data][1]["powersystem"]["baseMVA"] for j=1:num_gen if !(j in hydro_idx)) for i=1:num_sim) for t=1:num_stages-rm_stages]

savefig(plot(thermal_gen, legend=false, xlabel="Stage", ylabel="Mwh", title="Thermal-Generation $(case)-$(formulation_b)-$(formulation)")
, joinpath(case_dir, string(formulation), "SDDP-$(case)-$(formulation_b)-$(formulation)-thermal.png")
)

df = CSV.read(joinpath(case_dir, string(formulation), "MeanGeneration.csv"), DataFrame)
df[!, "$(string(formulation_b))"] = thermal_gen
# df = DataFrame(sddp=thermal_gen)

CSV.write(joinpath(case_dir, string(formulation), "MeanGeneration.csv"), df)

savefig(plot(Matrix(df[!,labels]), labels=permutedims(names(df[!,labels])), 
    xlabel="Stage", ylabel="Thermal Generation (MWh)", color=colors, 
    shape=markers, ylim=(minimum(Matrix(df))*0.9, maximum(Matrix(df))*1.1))
, joinpath(case_dir, string(formulation), "Comparison-$(case)-thermal.png")
)


# ## Objective
objective_values = [sum(results[:simulations][i][t][:stage_objective] for t=1:num_stages-rm_stages) for i=1:length(results[:simulations])]
println("Mean Sim: ", mean(objective_values))
