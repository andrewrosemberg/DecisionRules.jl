using DecisionRules
using Test
using HiGHS
import ParametricOptInterface as POI
using JuMP
using Zygote
using Flux
using Random

@testset "DecisionRules.jl" begin
    d = 10
    model = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_attributes(model, "output_flag" => false)
    
    @variable(model, x >= 0.0)
    @variable(model, 0.0 <= y <= 8.0)
    @variable(model, 0.0 <= state_out_var <= 8.0)
    @variable(model, _deficit)
    @variable(model, norm_deficit)
    @variable(model, state_in in MOI.Parameter(5.0))
    @variable(model, state_out in MOI.Parameter(4.0))
    @variable(model, uncertainty in MOI.Parameter(2.0))
    @constraint(model, state_out_var == state_in + uncertainty - x)
    @constraint(model, x + y >= d)
    @constraint(model, _deficit == state_out_var - state_out)
    @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(2))
    @objective(model, Min, 30 * y + norm_deficit * 10^7)

    optimize!(model)

    @testset "pdual" begin
        @test DecisionRules.pdual(state_in) ≈ -30.0
        @test DecisionRules.pdual(state_out) ≈ 30.0
    end

    @testset "simulate_stage" begin
        inflow = 2.0
        state_param_in = [state_in]
        state_param_out = [(state_out, state_out_var)]
        uncertainty_sample = Dict(uncertainty => inflow)
        state_in_val = [5.0]
        state_out_val = [4.0]

        @test DecisionRules.simulate_stage(model, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val) ≈ 210
        grad = gradient(DecisionRules.simulate_stage, model, state_param_in, state_param_out, uncertainty_sample, state_in_val, state_out_val)
        @test grad[5] ≈ [-30.0]
        @test grad[6] ≈ [30.0]
        subproblem = model
        # seed
        Random.seed!(222)
        m = Chain(Dense(1, 10), Dense(10, 1))
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) > 90.0
        for _ in 1:2050
            _inflow = rand(1.:5)
            uncertainty_samp = Dict(uncertainty => _inflow)
            Flux.train!((m, inflow) -> DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m(inflow)), m, [[_inflow] for _ =1:10], Flux.Adam())
        end
        @test DecisionRules.simulate_stage(subproblem, state_param_in, state_param_out, uncertainty_sample, state_in_val, m([inflow])) <= 92
    end
end
