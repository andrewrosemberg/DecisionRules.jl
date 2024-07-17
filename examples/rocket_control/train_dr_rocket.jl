using Flux
using DecisionRules
using Random
using Statistics

using JuMP
import Ipopt, HSL_jll

include("./examples/rocket_control/build_rocket_problem.jl")

det_equivalent, state_params_in, state_params_out, final_state, uncertainty_samples, x_v, x_h, x_m, u_t_max = build_rocket_problem(; penalty=1e-5)

# Create ML policy to solve the problem
models = Chain(Dense(1, 32, sigmoid), LSTM(32, 32), Dense(32, 1, (x) -> sigmoid(x) .* u_t_max))

# Pre-train

Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    final_state, sample(uncertainty_samples), 
    models;
) for _ in 1:2]
best_obj = mean(objective_values)

example_dir = joinpath(pwd(), "examples", "rocket_control") #dirname(@__FILE__)

model_dir = joinpath(example_dir, "models")
save_file = "rocket_model"
mkpath(model_dir)

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(best_obj, model_path, 0.003)

train_multistage(models, final_state, det_equivalent, state_params_in, state_params_out, uncertainty_samples; 
    num_batches=10,
    num_train_per_batch=1,
    optimizer=Flux.Adam(),
    record_loss= (iter, model, loss, tag) -> begin
        if tag == "metrics/training_loss"
            save_control(iter, model, loss)
        end
        println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
)


#####################################################################


# Finally, we plot the solution:

# function plot_trajectory(y; kwargs...)
#     return Plots.plot(
#         (1:T) * Δt,
#         value.(y);
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

