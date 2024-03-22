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
