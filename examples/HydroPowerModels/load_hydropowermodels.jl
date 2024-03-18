using JuMP
using CSV
using Tables
using JSON

function find_reservoirs_and_inflow(model::JuMP.Model)
    all_vars = all_variables(model)
    reservoir_in = all_vars[findall(x -> occursin("reservoir", name(x)) && !occursin("in", name(x)), all_vars)]
    reservoir_in = [all_vars[findfirst(x -> "reservoir[$i]_in" == name(x), all_vars)] for i in 1:length(reservoir_in)]
    reservoir_out = all_vars[findall(x -> occursin("reservoir", name(x)) && occursin("out", name(x)), all_vars)]
    reservoir_out = [all_vars[findfirst(x -> "reservoir[$i]_out" == name(x), all_vars)] for i in 1:length(reservoir_out)]
    inflow = all_vars[findall(x -> occursin("inflow", name(x)), all_vars)]
    inflow = [all_vars[findfirst(x -> "inflow[$i]" == name(x), all_vars)] for i in 1:length(reservoir_in)]
    return reservoir_in, reservoir_out, inflow
end

# function move_bounds_to_constrainits!(variable::JuMP.variableiableRef)
#     model = JuMP.owner_model(variable)
#     if has_lower_bound(variable)
#         @constraint(model, variable >= lower_bound(variable))
#         delete_lower_bound(variable)
#     end
#     if has_upper_bound(variable)
#         @constraint(model, variable <= upper_bound(variable))
#         delete_upper_bound(variable)
#     end
# end

function variable_to_parameter(model::JuMP.Model, variable::JuMP.VariableRef; initial_value=0.0)
    parameter = @variable(model; base_name = "_" * name(variable), set=MOI.Parameter(initial_value))
    # bind the parameter to the variable
    @constraint(model, variable == parameter)
    return parameter
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

function read_inflow(file::String, nHyd::Int; num_stages=nothing)
    allinflows = CSV.read(file, Tables.matrix; header=false)
    nlin, ncol = size(allinflows)
    if isnothing(num_stages)
        num_stages = nlin
    elseif num_stages > nlin
        number_of_cycles = div(num_stages, nlin) + 1
        allinflows = vcat([allinflows for _ in 1:number_of_cycles]...)
    end
    nCen = Int(floor(ncol / nHyd))
    vector_inflows = Array{Array{Float64,2}}(undef, nHyd)
    for i in 1:nHyd
        vector_inflows[i] = allinflows[1:num_stages, ((i - 1) * nCen + 1):(i * nCen)]
    end
    return vector_inflows, nCen, num_stages
end

function build_hydropowermodels(case_folder::AbstractString, subproblem_file::AbstractString; num_stages=nothing)
    hydro_file = JSON.parsefile(joinpath(case_folder, "hydro.json"))["Hydrogenerators"]
    nHyd = length(hydro_file)
    vector_inflows, nCen, num_stages = read_inflow(joinpath(case_folder, "inflows.csv"), nHyd; num_stages=num_stages)
    initial_state = [hydro["initial_volume"] for hydro in hydro_file]
    max_volume = [hydro["max_volume"] for hydro in hydro_file]

    subproblems = Vector{JuMP.Model}(undef, num_stages)
    state_params_in = Vector{Vector{VariableRef}}(undef, num_stages)
    state_params_out = Vector{Vector{VariableRef}}(undef, num_stages)
    uncertainty_samples = Vector{Dict{VariableRef, Vector{Float64}}}(undef, num_stages)
    
    for t in 1:num_stages
        subproblems[t] = JuMP.read_from_file(joinpath(case_folder, subproblem_file))
        state_params_in[t], state_params_out[t], inflow = find_reservoirs_and_inflow(subproblems[t])
        # move_bounds_to_constrainits!.(state_params_in[t])
        # move_bounds_to_constrainits!.(state_params_out[t])
        # move_bounds_to_constrainits!.(inflow)
        state_params_in[t] = variable_to_parameter.(subproblems[t], state_params_in[t])
        state_params_out[t] = variable_to_parameter.(subproblems[t], state_params_out[t])
        inflow = variable_to_parameter.(subproblems[t], inflow)
        add_deficit_constraints!(subproblems[t])
        uncertainty_dict = Dict{VariableRef, Vector{Float64}}()
        for (i, inflow_var) in enumerate(inflow)
            uncertainty_dict[inflow_var] = vector_inflows[i][t, :]
        end
        uncertainty_samples[t] = uncertainty_dict
    end

    return subproblems, state_params_in, state_params_out, uncertainty_samples, initial_state, max_volume
end

function ensure_feasibility(state_out, state_in, uncertainty, max_volume)
    state_out = max.(state_out, 0)
    state_out = min.(state_out, state_in .+ uncertainty)
    state_out = min.(state_out, max_volume)
    return state_out
end

function ensure_feasibility_sigmoid(state_out, state_in, uncertainty, max_volume)
    return sigmoid.(state_out) .* min.(max_volume, state_in .+ uncertainty)
end