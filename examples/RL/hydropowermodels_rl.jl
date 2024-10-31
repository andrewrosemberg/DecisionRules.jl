import QuickPOMDPs: QuickPOMDP
import POMDPTools: ImplicitDistribution, Deterministic
import Distributions: Normal, Uniform
using POMDPs
using Flux
using Crux
import Crux: state_space
using POMDPs
import POMDPTools:FunctionPolicy
using Random
using Distributions

using DecisionRules
using CommonRLInterface
using CommonRLSpaces

using Ipopt, HSL_jll

HydroPowerModels_dir = "./examples/HydroPowerModels"#dirname(@__FILE__)
include(joinpath(HydroPowerModels_dir, "load_hydropowermodels.jl"))

case_name = "bolivia" # bolivia, case3
formulation = "ACPPowerModel" # SOCWRConicPowerModel, DCPPowerModel, ACPPowerModel
num_stages = 96 # 96, 48

formulation_file = formulation * ".mof.json"

subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume = build_hydropowermodels(    
    joinpath(HydroPowerModels_dir, case_name), formulation_file; num_stages=num_stages, param_type=:Var
)

for subproblem in subproblems
    set_optimizer(subproblem, optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,
        # "hsllib" => HSL_jll.libhsl_path,
        # "linear_solver" => "ma27"
    ))
end

Random.seed!(1234)
uncertainty_sample = DecisionRules.sample(uncertainty_samples[1])
# split.(name.(keys(uncertainty_sample)), ["[", "]"])
idx = [parse(Int, split(split(i, "[")[2], "]")[1]) for i in name.(keys(uncertainty_sample))]
rain_state = collect(values(uncertainty_sample))[idx]

num_a = length(state_params_in[1])
mdp = QuickPOMDP(
    actions = Box(zeros(num_a), max_volume),
    obstype = Array{Float64,1},
    # discount = 0.995,

    gen = function (state, state_out, rng)
        state_out = sigmoid.(state_out .+ 1.0) .* max_volume
        state_in, rain_state, j = state[1:num_a], state[num_a+1:end-1], ceil(Int, state[end])
        uncertainty_sample = if j == num_stages
            Dict{Any, Float64}(DecisionRules.sample(uncertainty_samples[j]))
        else
            Dict{Any, Float64}(DecisionRules.sample(uncertainty_samples[j+1]))
        end
        rain = collect(values(uncertainty_sample))[idx]
        kys = keys(uncertainty_sample)
        for (i, ky) in enumerate(kys)
            uncertainty_sample[ky] = rain_state[i]
        end
        r = simulate_stage(subproblems[j], state_params_in[j], state_params_out[j], uncertainty_sample, state_in, state_out)
        sp = DecisionRules.get_next_state(subproblems[j], state_params_out[j], state_in, state_out)
        @info "Stage t=$j" sum(state_in) sum(rain_state) sum(state_out) sum(sp) r
        sp = [sp; rain; j+1]
        o = sp # no hidden state
        return (sp=sp, o=o, r=r)
    end,

    initialstate = Deterministic([initial_state; rain_state; 1.0]),
    initialobs = (s) -> Deterministic(s),
    isterminal = s -> s[end] > num_stages
)

amin = zeros(num_a)
amax = max_volume

rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

S = state_space(mdp)
V() = ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
SG() = SquashedGaussianPolicy(ContinuousNetwork(Chain(Dense(2*num_a+1, 64, relu), Dense(64, 64, relu), Dense(64, num_a, tanh))), zeros(Float32, 1), 1f0)

# Solve with REINFORCE
𝒮_reinforce = REINFORCE(π=SG(), S=S, N=2, ΔN=2, a_opt=(batch_size=2,))
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with PPO 
𝒮_ppo = PPO(π=ActorCritic(SG(), V()), S=S, N=100, ΔN=10, a_opt=(batch_size=2,), λe=0f0)
@time π_ppo = solve(𝒮_ppo, mdp)