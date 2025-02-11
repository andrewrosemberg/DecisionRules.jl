using JLD2
using SparseArrays

# include(joinpath(@__DIR__, "../mpc_utils.jl"))
include(joinpath(@__DIR__, "atlas_utils.jl"))

# Setup model and visualizer
atlas = Atlas();
vis = Visualizer();
mvis = init_visualizer(atlas, vis)

# Load balanced reference
@load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref;
visualize!(atlas, mvis, x_ref)

# Calculate discrete dynamics for a balanced position
# h = 0.01;
# Ad = FD.jacobian(x->rk4(atlas, x, u_ref, h), x_ref);
# Bd = FD.jacobian(u->rk4(atlas, x_ref, u, h), u_ref);

function memoize(foo::Function, n_outputs::Int)
    last_x, last_f = nothing, nothing
    last_dx, last_dfdx = nothing, nothing
    function foo_i(i, x::T...) where {T<:Real}
        if T == Float64
            if x !== last_x
                last_x, last_f = x, foo(x...)
            end
            return last_f[i]::T
        else
            if x !== last_dx
                last_dx, last_dfdx = x, foo(x...)
            end
            return last_dfdx[i]::T
        end
    end
    return [(x...) -> foo_i(i, x...) for i in 1:n_outputs]
end

function atlas_dynamics(xu::T...) where {T<:Real}
    h = 0.01
    x = collect(xu[1:atlas.nx])
    u = collect(xu[atlas.nx+1:end])
    return rk4(atlas, x, u, h)
end

# Lets define a jump model that solves the MPC problem
# min ∑_t ||x_t - x_ref||_2^2
# s.t. x_{t+1} = rk4(x_t, u_t) : this needs to be a jump operator for which we will need to define a method to calculate the jacobian and hessian
#      u_min <= u_t <= u_max

using JuMP
using Ipopt

model = Model(Ipopt.Optimizer)

# variables
N=2
@variable(model, x[t=1:N,1:atlas.nx])
@variable(model, -atlas.torque_limits[i] <= u[t=1:N-1,i=1:atlas.nu] <= atlas.torque_limits[i])

# objective
@objective(model, Min, sum(sum((x[t,:] - x_ref).^2 for t=2:N)))

# dynamics
memoized_f = [memoize(atlas_dynamics, atlas.nx + atlas.nu) for i in 1:N-1]

for t=2:N,i in 1:atlas.nx
    op_dy = add_nonlinear_operator(model, atlas.nx + atlas.nu, memoized_f[t-1][i], 
        (g, xu...) -> ForwardDiff.gradient!(g, y -> memoized_f[t-1][i](y...), collect(xu)),
        name = Symbol("op_dy_$(t)_$i")
    )
    @constraint(model, x[t,i] == op_dy([x[t-1,:];u[t-1,:]]...))
end

# initial condition
@constraint(model, x[1,:] .== x_ref)
@constraint(model, x[1,atlas.nq + 5] == 1.3)

# solve
optimize!(model)


# # Set up cost matrices (hand-tuned)
# Q = spdiagm([1e3*ones(12); repeat([1e1; 1e1; 1e3], 3); 1e1*ones(8); 1e2*ones(12); repeat([1; 1; 1e2], 3); 1*ones(8)]);
# R = spdiagm(1e-3*ones(atlas.nu));

# # Calculate infinite-horizon LQR cost-to-go and gain matrices
# K, Qf = ihlqr(Ad, Bd, Q, R, Q, max_iters = 1000);

# # Define additional constraints for the QP (just torques for Atlas)
# horizon = 2;
# A_torque = kron(I(horizon), [I(atlas.nu) zeros(atlas.nu, atlas.nx)]);
# l_torque = repeat(-atlas.torque_limits - u_ref, horizon);
# u_torque = repeat(atlas.torque_limits - u_ref, horizon);

# # Setup QP
# H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_torque, l_torque, u_torque, K);

# # Setup solver
# m = ReLUQP.setup(H, g, A, l, u, verbose = false, eps_primal=1e-2, eps_dual=1e-2, max_iters=10, iters_btw_checks=1);

# # Simulate
# N = 300;
# X = [zeros(atlas.nx) for _ = 1:N];
# U = [zeros(atlas.nu) for _ = 1:N];
# X[1] = deepcopy(x_ref);
# X[1][atlas.nq + 5] = 1.3; # Perturb i.c.

# # Warmstart solver
# Δx = X[1] - x_ref;
# ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx);
# m.opts.max_iters = 4000;
# m.opts.check_convergence = false;
# ReLUQP.solve(m);
# m.opts.max_iters = 10;

# # Run simulation
# for k = 1:N - 1
#     # Get error
#     global Δx = X[k] - x_ref

#     # Update solver
#     ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)

#     # Solve and get controls
#     results = ReLUQP.solve(m)
#     global U[k] = results.x[1:atlas.nu] - K*Δx

#     # Integrate
#     global X[k + 1] = rk4(atlas, X[k], clamp.(u_ref + U[k], -atlas.torque_limits, atlas.torque_limits), h)
# end
# animate!(atlas, mvis, X, Δt=h);
# readline()