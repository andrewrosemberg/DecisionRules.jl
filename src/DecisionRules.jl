module DecisionRules

using JuMP
import ParametricOptInterface as POI
using Flux
using ChainRulesCore
import ChainRulesCore.rrule

export simulate_multistage, sample, train_multistage, simulate_states, simulate_stage

include("simulate_multistage.jl")

end
