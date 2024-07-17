using Flux
using DecisionRules
using Random
using Statistics

using JuMP
import Ipopt, HSL_jll
# import Plots


# All parameters in this model have been normalized to be dimensionless, and
# they are taken from [COPS3](https://www.mcs.anl.gov/~more/cops/cops3.pdf).
function build_rocket_problem(;
    h_0 = 1,                      # Initial height
    v_0 = 0,                      # Initial velocity
    m_0 = 1.0,                    # Initial mass
    m_T = 0.6,                    # Final mass
    g_0 = 1,                      # Gravity at the surface
    h_c = 500,                    # Used for drag
    c = 0.5 * sqrt(g_0 * h_0),    # Thrust-to-fuel mass
    D_c = 0.5 * 620 * m_0 / g_0,  # Drag scaling
    u_t_max = 3.5 * g_0 * m_0,    # Maximum thrust
    T = 1_000,                    # Number of time steps
    Δt = 0.2 / T,                 # Time per discretized step
    penalty = 10,                 # Penalty for violating target
    final_u_state = 0.0,          # Final state of the control
    num_scenarios = 10,           # Number of samples
)
    # ## JuMP formulation

    # First, we create a model and choose an optimizer. Since this is a nonlinear
    # program, we need to use a nonlinear solver like Ipopt. We cannot use a linear
    # solver like HiGHS.

    det_equivalent = Model()
    set_optimizer(det_equivalent, optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ))

    # Next, we create our state and control variables, which are each indexed by
    # `t`. It is good practice for nonlinear programs to always provide a starting
    # solution for each variable.

    @variable(det_equivalent, x_v[1:T], start = v_0)           # Velocity
    @variable(det_equivalent, x_h[1:T] >= 0, start = h_0)           # Height
    @variable(det_equivalent, x_m[1:T] >= m_T, start = m_0)         # Mass
    @variable(det_equivalent, 0 <= u_t[1:T] <= u_t_max, start = 0); # Thrust
    @variable(det_equivalent, target[1:T-1], start = 0);           # Thrust target
    @variable(det_equivalent, w[1:T-1] ∈ MOI.Parameter.(1.0));      # Wind
    @variable(det_equivalent, norm_deficit >= 0);                   # Wind

    # We implement boundary conditions by fixing variables to values.

    fix(x_v[1], v_0; force = true)
    fix(x_h[1], h_0; force = true)
    fix(x_m[1], m_0; force = true)
    fix(u_t[T], final_u_state; force = true)

    # The objective is to maximize altitude at end of time of flight.
    @constraint(det_equivalent, [norm_deficit; (target.-u_t[1:T-1])] in MOI.NormOneCone(T))

    @objective(det_equivalent, Min, -x_h[T] + penalty * norm_deficit)


    # Forces are defined as functions:

    D(x_h, x_v) = D_c * x_v^2 * exp(-h_c * (x_h - h_0) / h_0)
    g(x_h) = g_0 * (h_0 / x_h)^2

    # The dynamical equations are implemented as constraints.

    ddt(x::Vector, t::Int) = (x[t] - x[t-1]) / Δt
    @constraint(det_equivalent, [t in 2:T], ddt(x_h, t) == x_v[t-1])
    @constraint(
        det_equivalent,
        [t in 2:T],
        ddt(x_v, t) == (u_t[t-1] - D(x_h[t-1], x_v[t-1])) / x_m[t-1] - g(x_h[t-1]) - w[t-1],
    )
    @constraint(det_equivalent, [t in 2:T], ddt(x_m, t) == -u_t[t-1] / c)

    # uncertainty
    uncertainty_samples = Vector{Dict{Any, Vector{Float64}}}(undef, T-1)
    for t in 1:T-1
        uncertainty_dict = Dict{Any, Vector{Float64}}()
        uncertainty_dict[w[t]] = randn(num_scenarios)
        uncertainty_samples[t] = uncertainty_dict
    end

    return det_equivalent, vcat([VariableRef[u_t[T]]], [VariableRef[i] for i in u_t[1:T-2]]), [[(target[t], u_t[t])] for t in 1:T-1], [final_u_state], uncertainty_samples, x_v, x_h, x_m, u_t_max
end

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

# load model
using JLD2
nn = Chain(Dense(1, 32, sigmoid), LSTM(32, 32), Dense(32, 1, (x) -> sigmoid(x) .* u_t_max))
opt_state = Flux.setup(Flux.Adam(), nn)
x = randn(1, 1)
y = rand(1, 1)
train_set = [(x, y)]
Flux.train!(nn, train_set, opt_state) do m, x, y
    Flux.mse(m(x), y)
end
model_state = JLD2.load(model_path, "model_state")
Flux.loadmodel!(nn, model_state)

# simulate
Random.seed!(8788)
objective_values = [simulate_multistage(
    det_equivalent, state_params_in, state_params_out, 
    final_state, sample(uncertainty_samples), 
    nn;
    _objective_value = DecisionRules.get_objective_no_target_deficit
) for _ in 1:2]
best_obj = mean(objective_values)


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

Plots.plot(
    plot_trajectory(x_h; ylabel = "Altitude"),
    plot_trajectory(x_m; ylabel = "Mass"),
    plot_trajectory(x_v; ylabel = "Velocity"),
    plot_trajectory(u_t; ylabel = "Thrust");
    layout = (2, 2),
)

