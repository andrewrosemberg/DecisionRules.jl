function variable_to_parameter(model::JuMP.Model, variable::JuMP.VariableRef; initial_value=0.0, deficit=nothing)
    parameter = @variable(model; base_name = "_" * name(variable), set=MOI.Parameter(initial_value))
    # bind the parameter to the variable
    if isnothing(deficit)
        @constraint(model, variable == parameter)
        return parameter
    else
        @constraint(model, variable + deficit == parameter)
        return parameter, variable
    end
end

function create_deficit!(model::JuMP.Model, len::Int; penalty=nothing)
    if isnothing(penalty)
        obj = objective_function(model)
        # get the highest coefficient
        penalty = maximum(abs.(values(obj.terms)))
        penalty = penalty * 1.1
    end
    _deficit = @variable(model, _deficit[1:len])
    @variable(model, norm_deficit)
    @constraint(model, [norm_deficit; _deficit] in MOI.NormOneCone(1 + len))
    set_objective_coefficient(model, norm_deficit, penalty)
    return norm_deficit, _deficit
end

mutable struct SaveBest <: Function
    best_loss::Float64
    model_path::String
    threshold::Float64
end
function (callback::SaveBest)(iter, model, loss)
    if loss < callback.best_loss
        @info "best model change" callback.best_loss loss
        callback.best_loss = loss
        model_state = Flux.state(model)
        jldsave(callback.model_path; model_state=model_state)
    end
    if loss < callback.threshold
        return true
    end
    return false
end

function var_set_name!(src::JuMP.VariableRef, dest::JuMP.VariableRef, t::Int)
    name = JuMP.name(src)
    if !isempty(name)
        # append node index to original variable name
        JuMP.set_name(dest, string(name, "#", t))
    else
        # append node index to original variable index
        var_name = string("_[", index(src).value, "]")
        JuMP.set_name(dest, string(var_name, "#", t))
    end
end

function add_child_model_vars!(model::JuMP.Model, subproblem::JuMP.Model, t::Int, state_params_in::Vector{Vector{VariableRef}}, state_params_out::Vector{Vector{Tuple{VariableRef, VariableRef}}}, initial_state::Vector{Float64}, var_src_to_dest::Dict{VariableRef, VariableRef})
    allvars = all_variables(subproblem)
    allvars = setdiff(allvars, state_params_in[t])
    allvars = setdiff(allvars, [x[1] for x in state_params_out[t]])
    allvars = setdiff(allvars, [x[2] for x in state_params_out[t]])
    x = @variable(model, [1:length(allvars)])
    for (src, dest) in zip(allvars, x)
        var_src_to_dest[src] = dest
        var_set_name!(src, dest, t)
    end
    st_out_param = @variable(model, [1:length(state_params_out[t])])
    st_out_var = @variable(model, [1:length(state_params_out[t])])
    for (i, src) in enumerate(state_params_out[t])
        dest_param = @variable(model)
        dest_var = @variable(model)
        var_src_to_dest[src[1]] = dest_param
        var_src_to_dest[src[2]] = dest_var
        var_set_name!(src[1], dest_param, t)
        var_set_name!(src[2], dest_var, t)
        state_params_out[t][i] = (dest_param, dest_var)
    end
    if t == 1
        for (i, src) in enumerate(state_params_in[t])
            dest = @variable(model)
            var_src_to_dest[src] = dest
            var_set_name!(src, dest, t)
            state_params_in[t][i] = dest
        end
    else
        for (i, src) in enumerate(state_params_in[t])
            var_src_to_dest[src] = state_params_out[t-1][i][2]
            state_params_in[t][i] = state_params_out[t-1][i][2]
            # delete parameter constraint associated with src
            for con in JuMP.all_constraints(subproblem, VariableRef, MOI.Parameter{Float64})
                obj = JuMP.constraint_object(con)
                if obj.func == src
                    JuMP.delete(subproblem, con)
                end
            end
        end
    end
    return var_src_to_dest
end

function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a],
                src_to_dest_variable[pair.b],
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::Any,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return throw_detequiv_error(
        "`copy_and_replace_variables` is not implemented for functions like `$(src)`.",
    )
end

function add_child_model_exps!(model::JuMP.Model, subproblem::JuMP.Model, var_src_to_dest::Dict{VariableRef, VariableRef})
    # Add constraints:
    for (F, S) in JuMP.list_of_constraint_types(subproblem)
        for con in JuMP.all_constraints(subproblem, F, S)
            obj = JuMP.constraint_object(con)
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model, new_func in obj.set)
        end
    end
    # Add objective:
    current = JuMP.objective_function(model)
    subproblem_objective =
        copy_and_replace_variables(JuMP.objective_function(subproblem), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + subproblem_objective,
    )
end

"Create Single JuMP.Model from subproblems. rename variables to avoid conflicts by adding [t] at the end of the variable name where t is the subproblem index"
function deterministic_equivalent(subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{Tuple{VariableRef, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Vector{Float64}}},
)
    model = JuMP.Model()
    set_objective_sense(model, objective_sense(subproblems[1]))
    uncertainties_new = Vector{Dict{VariableRef, Vector{Float64}}}(undef, length(uncertainties))
    var_src_to_dest = Dict{VariableRef, VariableRef}()
    for t in 1:length(subproblems)
        DecisionRules.add_child_model_vars!(model, subproblems[t], t, state_params_in, state_params_out, initial_state, var_src_to_dest)
        uncertainties_new[t] = Dict{VariableRef, Vector{Float64}}()
        for (ky, val) in uncertainties[t]
            uncertainties_new[t][var_src_to_dest[ky]] = val
        end
    end

    for t in 1:length(subproblems)
        DecisionRules.add_child_model_exps!(model, subproblems[t], var_src_to_dest)
    end

    return model, uncertainties_new
end