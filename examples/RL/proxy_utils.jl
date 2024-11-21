function get_standard_form_data(subproblem::JuMP.Model, state_param_in::Vector{Any}, state_param_out::Vector{Tuple{Any, VariableRef}}, uncertainty::Dict{Any, T}, state_in::Vector{Z}, state_out_target::Vector{V}
    ) where {T <: Real, V <: Real, Z <: Real}
    # Update state parameters
    for (i, state_var) in enumerate(state_param_in)
        set_parameter(subproblem, state_var, state_in[i])
    end

    # Update uncertainty
    for (uncertainty_param, uncertainty_value) in uncertainty
        set_parameter(subproblem, uncertainty_param, uncertainty_value)
    end

    # Update state parameters out
    for i in 1:length(state_param_in)
        state_var = state_param_out[i][1]
        set_parameter(subproblem, state_var, state_out_target[i])
    end

    std = make_standard_form_data(subproblem)

    return std
end

function dual_objective(y, A, b, c, l, u)
    zl, zu = zlzu_from_y(y, c, A)
    return sum(b .* y) + sum(l .* zl) - sum(u .* zu)
end

function zlzu_from_y(y, c, A)
    z = c - A' * y
    return max(0, z), max(0, -z)
end