using Distributed
using Random

@everywhere l2o_path = @__DIR__

@everywhere import Pkg

@everywhere Pkg.activate(l2o_path)

@everywhere Pkg.instantiate()

@everywhere import QuickPOMDPs: QuickPOMDP
@everywhere import POMDPTools: ImplicitDistribution, Deterministic
@everywhere import Distributions: Normal, Uniform
@everywhere using POMDPs
@everywhere using Flux
@everywhere using Crux
@everywhere import Crux: state_space
@everywhere using POMDPs
@everywhere import POMDPTools:FunctionPolicy
@everywhere using Random
@everywhere using Distributions

@everywhere using DecisionRules
@everywhere using CommonRLInterface
@everywhere using CommonRLSpaces

@everywhere using Ipopt, HSL_jll
#"./examples/HydroPowerModels"#dirname(@__FILE__)
@everywhere HydroPowerModels_dir =joinpath(dirname(l2o_path), "HydroPowerModels")
@everywhere include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

@everywhere case_name = "bolivia" # bolivia, case3
@everywhere formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
@everywhere num_stages = 96 # 96, 48

@everywhere function build_mdp(case_name, formulation, num_stages; solver=optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        "hsllib" => HSL_jll.libhsl_path,
        "linear_solver" => "ma27"
    ),
    penalty=nothing
)
    formulation_file = formulation * ".mof.json"
    subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
        joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages, param_type=:Var, penalty=penalty
    )

    for subproblem in subproblems
        set_optimizer(subproblem, solver)
    end

    # test solve
    JuMP.optimize!(subproblems[1])
    termination_status(subproblems[1])

    Random.seed!(1234)
    uncertainty_sample = DecisionRules.sample(uncertainty_samples)
    # split.(name.(keys(uncertainty_sample)), ["[", "]"])
    uncertainty_samples_vec = [collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)]
    idx = [parse(Int, split(split(i, "[")[2], "]")[1]) for i in name.(keys(uncertainty_sample[1]))]
    rain_state = uncertainty_samples_vec[1]

    num_a = length(state_params_in[1])
    mdp = QuickPOMDP(
        actions = Box(zeros(num_a), max_volume),
        obstype = Array{Float64,1},
        # discount = 0.995,

        gen = function (state, state_out, rng)
            # Scale the normalized policy output to the action space
            state_out = sigmoid.(state_out .+ 1.0) .* max_volume
            state_in, rain_state, j = state[1:num_a], state[num_a+1:end-1], ceil(Int, state[end])
            rain = if j == num_stages
                uncertainty_samples_vec[j]
            else
                uncertainty_samples_vec[j+1]
            end
            r = simulate_stage(subproblems[j], state_params_in[j], state_params_out[j], Dict{Any, Float64}(uncertainty_sample[j]), state_in, state_out)
            sp = DecisionRules.get_next_state(subproblems[j], state_params_out[j], state_in, state_out)
            @info "Stage t=$j" sum(state_in) sum(rain_state) sum(state_out) sum(sp) r
            sp = [sp; rain; j+1]
            o = sp # no hidden state
            return (sp=sp, o=o, r=-r)
        end,

        initialstate = Deterministic([initial_state; rain_state; 1.0]),
        initialobs = (s) -> Deterministic(s),
        isterminal = s -> s[end] > num_stages
    )
    return mdp, num_a, max_volume
end
# amin = zeros(num_a)
# amax = max_volume

# rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

@everywhere mdp, num_a, max_volume = build_mdp(case_name, formulation, num_stages)

@everywhere S = state_space(mdp)

# Define the networks we will use
@everywhere QSA() = ContinuousNetwork(Chain(Dense(34, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
@everywhere V() = ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
@everywhere A() = ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, num_a, tanh)))
@everywhere SG() = SquashedGaussianPolicy(ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, num_a, tanh))), zeros(Float32, 1), 1f0)

# build the MDPs

# Solve with REINFORCE
@everywhere 𝒮_reinforce = REINFORCE(π=SG(), S=S, N=1000, ΔN=10, a_opt=(batch_size=10,))
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with PPO 
@everywhere 𝒮_ppo = PPO(π=ActorCritic(SG(), V()), S=S, N=10000, ΔN=10, a_opt=(batch_size=10,))
@time π_ppo = solve(𝒮_ppo, mdp)

# Off-policy settings
@everywhere off_policy = (S=S,
              ΔN=10,
              N=10000,
              buffer_size=Int(200),
              buffer_init=200,
              c_opt=(batch_size=128, optimizer=Adam(1e-3)),
              a_opt=(batch_size=128, optimizer=Adam(1e-3)),
              π_explore=GaussianNoiseExplorationPolicy(0.5f0, a_min=[0.0], a_max=[1.0])
)
              
# Solver with DDPG
@everywhere 𝒮_ddpg = DDPG(;π=ActorCritic(A(), QSA()), off_policy...)
@time π_ddpg = solve(𝒮_ddpg, mdp)

# Solve with TD3
@everywhere 𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_td3 = solve(𝒮_td3, mdp)

# Solve with SAC
@everywhere 𝒮_sac = SAC(;π=ActorCritic(SG(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)

# pmap(solve, [𝒮_reinforce, 𝒮_ppo, 𝒮_ddpg, 𝒮_td3, 𝒮_sac], [build_mdp(case_name, formulation, num_stages)[1] for _ in 1:5])

using Plots
function plot_learning_mod(input; 
        title = "",
        ylabel = "Undiscounted Return (Costs)",  
        xlabel = "Training Steps", 
        values = :undiscounted_return, 
        labels = :default,
        legend = :bottomright,
        font = :palatino,
        p = plot(), 
        vertical_lines = [],
        vline_range = (0, 1), 
        thick_every = 1,
        yscale = :log10,
        xs=nothing
    )
    dirs = directories(input)
    
    N = length(dirs)
    values isa Symbol && (values = fill(values, N))
    if labels == :default
        labels = N == 1 ? [""] : ["Task $i" for i=1:N]
    end 
    
    # Plot the vertical lines (usually for multitask learning or to designate a point on the curve)
    for i = 1:length(vertical_lines)
        plot!(p, [vertical_lines[i], vertical_lines[i]], [vline_range...], color=:black, linewidth = i % thick_every == 0 ? 3 : 1, label = "")
    end
    
    # Plot the learning curves
    plot!(p, ylabel = ylabel, xlabel = xlabel, legend = legend, title = title, fontfamily = font, yscale=yscale)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        if xs != nothing
            x = x[xs]
            y = y[xs]
        end
        plot!(p, x, -y, alpha = 0.3, color=i, label = "", yscale=yscale)
        plot!(p, x, smooth(-y), color=i, label = labels[i], linewidth =2, yscale=yscale)
    end
    p
end

p = plot_learning_mod(
    [
        𝒮_reinforce, 
        𝒮_ppo, 
        𝒮_ddpg,
        𝒮_td3,
        𝒮_sac,
    ], 
    title="LTHD Training Curves", 
    labels=[
        "REINFORCE", 
        "PPO", 
        "DDPG",
        "TD3",
        "SAC"
    ], legend=:outertopright, yscale=:log10,
)

using CSV, DataFrames
tsgdr_df = CSV.read("../../tsgdr_train.csv", DataFrame)

plot!(p, tsgdr_df[:,"Step"], tsgdr_df[:,"bolivia-ACPPowerModel-h96-2024-05-18T17:06:09.485 - metrics/training_loss"], label="TS-GDR", linewidth=2, yscale=:log10)

Crux.savefig("./hydro_benchmark.pdf")
