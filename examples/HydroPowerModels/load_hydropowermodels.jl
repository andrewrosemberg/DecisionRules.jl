using JuMP
using CSV
using DataFrames

function find_reservoirs_and_inflow(model::JuMP.Model)
    all_vars = all_variables(model)
    reservoir_in = all_vars[findall(x -> occursin("reservoir", name(x)) && !occursin("in", name(x)), all_vars)]
    reservoir_out = all_vars[findall(x -> occursin("reservoir", name(x)) && occursin("out", name(x)), all_vars)]
    inflow = all_vars[findall(x -> occursin("inflow", name(x)), all_vars)]
    inflow = [all_vars[findfirst(x -> "inflow[$i]" == name(x), all_vars)] for i in 1:length(reservoir_in)]
    return reservoir_in, reservoir_out, inflow
end

function move_bounds_to_constrainits!(var::JuMP.VariableRef)
    if has_lower_bound(var)
        @constraint(model, var >= lower_bound(var))
        delete_lower_bound(var)
    end
    if has_upper_bound(var)
        @constraint(model, var <= upper_bound(var))
        delete_upper_bound(var)
    end
end

function variable_to_parameter(model::JuMP.Model, var::JuMP.VariableRef; initial_value=0.0)
    return @constraint(model, var in MOI.Parameter(initial_value))
end

function add_deficit_constraints!(model::JuMP.Model; penalty=nothing)
    if isnothing(penalty)
        obj = objective_function(model)
        # get the highest coefficient
        penalty = maximum(abs.(values(obj.terms)))
        penalty = penalty * 1.1
    end
    consrefs = [con for con in all_constraints(model, include_variable_in_set_constraints=false)]
    @variable(model, _deficit[1:length(consrefs)])
    @variable(model, norm_deficit)
    for (i, eq) in enumerate(consrefs)
        set_normalized_coefficient(eq, _deficit[i], 1)
    end
    @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(1 + length(_deficit)))
    set_objective_coefficient(model, norm_deficit, penalty)
    return norm_deficit
end

function build_hydropowermodels(case_folder::AbstractString; num_stages=nothing)
    inflow_data = CSV.read(joinpath(case_folder, "inflows.csv"), DataFrame; header=false)
    if isnothing(num_stages)
        num_stages = size(inflow_data, 1)
    elseif num_stages > size(inflow_data, 1)
        number_of_cycles = div(num_stages, size(inflow_data, 1)) + 1
        inflow_data = vcat([inflow_data for _ in 1:number_of_cycles]...)
    end
    subproblems = Vector{JuMP.Model}(undef, num_stages)
    state_params_in = Vector{Vector{VariableRef}}(undef, num_stages)
    state_params_out = Vector{Vector{VariableRef}}(undef, num_stages)
    uncertainty_samples = Vector{Dict{VariableRef, Vector{Float64}}}(undef, num_stages)
    
    for t in 1:num_stages
        subproblems[t] = read_from_file(joinpath(@__DIR__, "DCPPowerModel.mof.json"))
        state_params_in[t], state_params_out[t], inflow = find_reservoirs_and_inflow(subproblems[t])
        move_bounds_to_constrainits!.(reservoir_in)
        move_bounds_to_constrainits!.(reservoir_out)
        move_bounds_to_constrainits!.(inflow)
        variable_to_parameter.(subproblems[t], reservoir_in)
        variable_to_parameter.(subproblems[t], reservoir_out)
        variable_to_parameter.(subproblems[t], inflow)
        add_deficit_constraints!(subproblems[t])
        
        uncertainty_samples[t] = Dict(inflow .=> inflow_data[t, :])
    end

    return subproblems, state_params_in, state_params_out, uncertainty_samples
end