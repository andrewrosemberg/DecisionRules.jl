import SparseArrays: SparseMatrixCSC, findnz
using PowerModels
using JuMP

const PM = PowerModels

struct StandardFormData
    A::SparseMatrixCSC
    b::Vector{Float64}
    c::Vector{Float64}
    c0::Float64
    l::Vector{Float64}
    u::Vector{Float64}
    columns::Dict{VariableRef, Union{Int, Dict{String, Float64}}}
end


"""
    make_standard_form_data(lp::Model)

Convert a JuMP Model to standard form matrices:

    min  ∑ᵢ cᵢxᵢ + c₀
    s.t.     Ax == b
         l <= x <= u

The input model must only have linear constraints (GenericAffExpr).
If the input model has a quadratic (QuadExpr) objective, only the linear parts are used.
"""
function make_standard_form_data(lp::Model)
    jump_std = JuMP._standard_form_matrix(lp)
    # this returns
    # [A -I] [x, y] = 0
    # [c_l, r_l] <= [x, y] <= [c_u, r_u]

    function deletecol(A, col)
        dim = length(size(A))
        if dim == 1
            A = [A[1:col-1]; A[col+1:end]]
        elseif dim == 2
            A = [A[:, 1:col-1] A[:, col+1:end]]
        else
            error("deletecol: only 1D and 2D arrays are supported")
        end
        return A
    end

    columns = Dict{VariableRef, Union{Int, Dict{String, Float64}}}()
    for (var, col) in jump_std.columns
        columns[var] = col
    end

    A = copy(jump_std.A)
    l = copy(jump_std.lower)
    u = copy(jump_std.upper)
    b = zeros(size(jump_std.A, 1))

    equal_bound_idxs = findall(l .== u)
    if length(equal_bound_idxs) > 0
        for eqb_idx in reverse(sort(equal_bound_idxs))
            x_rows = findall(A[:, eqb_idx] .!= 0)
            x_coeffs = A[x_rows, eqb_idx]
            fixed_val = l[eqb_idx]

            b[x_rows] .-=  (x_coeffs .* fixed_val)

            A = deletecol(A, eqb_idx)
            l = deletecol(l, eqb_idx)
            u = deletecol(u, eqb_idx)

            for (var, col) in columns
                if !isa(col, Int)
                    continue
                elseif col > eqb_idx
                    columns[var] = col - 1
                elseif col == eqb_idx
                    columns[var] = Dict{String, Float64}()
                    columns[var]["fixed"] = fixed_val
                end
            end
        end
    end

    n_x = size(A, 2)

    obj_sense = objective_sense(lp)
    obj_func = objective_function(lp)

    function get_c(obj_func::JuMP.AffExpr, n_x::Int)
        vars = collect(keys(obj_func.terms))
        coeffs = collect(values(obj_func.terms))

        c = zeros(n_x)
        c0 = obj_func.constant

        for (var, coeff) in zip(vars, coeffs)
            col = columns[var]
            if isa(col, Int)
                c[col] = coeff
            elseif isa(col, Dict) && haskey(col, "fixed")
                c0 += coeff * col["fixed"]
            else
                error("get_c: unexpected col")
            end
        end
        return c, c0
    end

    function get_c(obj_func::JuMP.QuadExpr, n_x::Int)
        return get_c(obj_func.aff, n_x)
    end

    c, c0 = get_c(obj_func, n_x)

    if obj_sense == JuMP.MOI.MAX_SENSE
        c = -c
        c0 = -c0
    end

    return StandardFormData(A, b, c, c0, l, u, columns)
end


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
    return max.(0, z), max.(0, -z)
end