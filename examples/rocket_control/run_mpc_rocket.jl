using JuMP
import Ipopt
import Random
import Ipopt, HSL_jll

include("./examples/rocket_control/build_rocket_problem.jl")

# We can now run the simulation:

example_dir = joinpath(pwd(), "examples", "rocket_control") #dirname(@__FILE__)
save_dir = joinpath(example_dir, "mpc_results")
mkpath(save_dir)

using CSV
using DataFrames
# create two files to store the results for each seed. Each row will be a different seed. 
# We will have 1 column per time period
# 1. mpc_h.csv: store hight (x_h) for each time period
df = DataFrame(zeros(0,1000), [Symbol("$i") for i in 1:1000])
df[!, :seed] = Int[]
CSV.write(joinpath(save_dir, "mpc_h.csv"), df)
# 2. mpc_u.csv: store thrust (u_t) for each time period
CSV.write(joinpath(save_dir, "mpc_u.csv"), df)

seeds = 1:10
for seed in seeds
    Random.seed!(seed)
    x_h, x_m, x_v, u_t = run_rolling_mpc_time()
    df_aux = DataFrame([x_h; seed]'[:,:], vcat([Symbol("$i") for i in 1:1000], [:seed]))
    CSV.write(joinpath(save_dir, "mpc_h.csv"), df_aux, append=true)
    df_aux = DataFrame([u_t; seed]'[:,:], vcat([Symbol("$i") for i in 1:1000], [:seed]))
    CSV.write(joinpath(save_dir, "mpc_u.csv"), df_aux, append=true)
end

# load the results

# df_h = CSV.read(joinpath(save_dir, "mpc_h.csv"), DataFrame)
# df_u = CSV.read(joinpath(save_dir, "mpc_u.csv"), DataFrame)

# Random.seed!(0)
# x_h, x_m, x_v, u_t = run_rolling_mpc_time()

# Finally, we plot the solution:
# import Plots
# function plot_trajectory(y; kwargs...)
#     return Plots.plot(
#         (1:T) * Δt,
#         y;
#         xlabel = "Time (s)",
#         legend = false,
#         kwargs...,
#     )
# end

# Plots.plot(
#     plot_trajectory(x_h; ylabel = "Altitude"),
#     plot_trajectory(x_m; ylabel = "Mass"),
#     plot_trajectory(x_v; ylabel = "Velocity"),
#     plot_trajectory(u_t; ylabel = "Thrust");
#     layout = (2, 2),
# )

