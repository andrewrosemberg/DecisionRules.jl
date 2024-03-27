function simulate_states(
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    decision_rule::F;
    ensure_feasibility=(x_out, x_in, uncertainty) -> x
) where {F}
    num_stages = length(uncertainties)
    states = Vector{Vector{Float64}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = collect(values(uncertainties[stage]))
        state_out = decision_rule(uncertainties_stage)
        states[stage + 1] = ensure_feasibility(state_out, states[stage], uncertainties_stage)
    end
    return states
end

function simulate_states(
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    decision_rules::Vector{F};
    ensure_feasibility=(x_out, x_in, uncertainty) -> x
) where {F}
    num_stages = length(uncertainties)
    states = Vector{Vector{Float64}}(undef, num_stages + 1)
    states[1] = initial_state
    for stage in 1:num_stages
        uncertainties_stage = vcat(initial_state, [collect(values(uncertainties[j])) for j in 1:stage]...)
        decision_rule = decision_rules[stage]
        state_out = decision_rule(uncertainties_stage)
        states[stage + 1] = ensure_feasibility(state_out, states[stage], collect(values(uncertainties[stage])))
    end
    return states
end

function simulate_stage(subproblem::JuMP.Model, state_param_in::Vector{VariableRef}, state_param_out::Vector{Tuple{VariableRef, VariableRef}}, uncertainty::Dict{VariableRef, T}, state_in::Vector{Z}, state_out_target::Vector{V}
) where {T <: Real, V <: Real, Z <: Real}
    # Update state parameters
    for (i, state_var) in enumerate(state_param_in)
        MOI.set(subproblem, POI.ParameterValue(), state_var, state_in[i])
    end

    # Update uncertainty
    for (uncertainty_param, uncertainty_value) in uncertainty
        MOI.set(subproblem, POI.ParameterValue(), uncertainty_param, uncertainty_value)
    end

    # Update state parameters out
    for i in 1:length(state_param_in)
        state_var = state_param_out[i][1]
        MOI.set(subproblem, POI.ParameterValue(), state_var, state_out_target[i])
    end

    # Solve subproblem
    optimize!(subproblem)

    # objective value
    obj = objective_value(subproblem)

    return obj
end

function get_next_state(subproblem::JuMP.Model, state_param_out::Vector{Tuple{VariableRef, VariableRef}}, state_in::Vector{T}, state_out_target::Vector{Z}) where {T <: Real, Z <: Real}
    state_out = [value(state_param_out[i][2]) for i in 1:length(state_param_out)]
    return state_out
end

# Define rrule of get_next_state
# This is simplified. This actual jacobian will require DiffOpt
function rrule(::typeof(get_next_state), subproblem, state_param_out, state_in, state_out_target)
    state_out = get_next_state(subproblem, state_param_out, state_in, state_out_target)
    function _pullback(Δstate_out)
        d_state_in = zeros(length(state_in))
        d_state_out = zeros(length(state_out_target))
        for i in 1:length(state_in)
            s_out_target = state_out_target[i]
            s_in = state_in[i]
            if s_out_target < s_in && s_out_target >= 0.0
                d_state_out[i] = Δstate_out[i]
            elseif s_out_target > s_in && s_out_target >= 0.0
                d_state_in[i] = Δstate_out[i]
            end
        end
        return (NoTangent(), NoTangent(), NoTangent(), d_state_in , d_state_out)
    end
    return state_out, _pullback
end

function get_objective_no_target_deficit(subproblem::JuMP.Model; norm_deficit::AbstractString="norm_deficit")
    obj = JuMP.objective_function(subproblem)
    objective_val = objective_value(subproblem)
    for term in obj.terms
        if name(term[1]) == norm_deficit
            objective_val -= term[2] * value(term[1])
        end
    end
    return objective_val
end

# define rrule of get_objective_no_target_deficit
function rrule(::typeof(get_objective_no_target_deficit), subproblem; norm_deficit="norm_deficit")
    objective_val = get_objective_no_target_deficit(subproblem, norm_deficit=norm_deficit)
    function _pullback(Δobjective_val)
        return (NoTangent(), NoTangent())
    end
    return objective_val, _pullback
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{Tuple{VariableRef, VariableRef}}},
    states::Vector{Vector{Float64}},
    uncertainties::Vector{Dict{VariableRef, Float64}};
    get_objective_no_target_deficit = get_objective_no_target_deficit
    )
    
    # Loop over stages
    objective_value = 0.0
    state_in = states[1]
    for stage in 1:length(subproblems)
        state_out = states[stage + 1]
        subproblem = subproblems[stage]
        state_param_in = state_params_in[stage]
        state_param_out = state_params_out[stage]
        uncertainty = uncertainties[stage]
        simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
        objective_value += get_objective_no_target_deficit(subproblem)
        state_in = get_next_state(subproblem, state_param_out, state_in, state_out)
    end
    
    # Return final objective value
    return objective_value
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{Tuple{VariableRef, VariableRef}}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    decision_rules;
    ensure_feasibility=(x_out, x_in, uncertainty) -> x,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    states = simulate_states(initial_state, uncertainties, decision_rules, ensure_feasibility=ensure_feasibility)
    return simulate_multistage(subproblems, state_params_in, state_params_out, states, uncertainties; get_objective_no_target_deficit=get_objective_no_target_deficit)
end

pdual(v::VariableRef) = MOI.get(JuMP.owner_model(v), POI.ParameterDual(), v)
pdual(vs::Vector{VariableRef}) = [pdual(v) for v in vs]

# Define rrule of simulate_stage
function rrule(::typeof(simulate_stage), subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    y = simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    function _pullback(Δy)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), pdual.(state_param_in) * Δy, pdual.([s[1] for s in state_param_out]) * Δy)
    end
    return y, _pullback
end

function sample(uncertainty_samples::Dict{VariableRef, Vector{Float64}})
    return Dict((k => v[rand(1:end)]) for (k, v) in uncertainty_samples)
end

sample(uncertainty_samples::Vector{Dict{VariableRef, Vector{Float64}}}) = [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]

function train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_sampler; 
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        uncertainty_samples_vec = [[collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)] for uncertainty_sample in uncertainty_samples]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        objective = 0.0
        eval_loss = 0.0
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                Flux.reset!(model)
                m(initial_state)
                state_in = initial_state
                for (j, subproblem) in enumerate(subproblems)
                    state_out = m(uncertainty_samples_vec[s][j])
                    state_out = ensure_feasibility(state_out, state_in, uncertainty_samples_vec[s][j])
                    objective += simulate_stage(subproblem, state_params_in[j], state_params_out[j], uncertainty_samples[s][j], state_in, state_out)
                    eval_loss += get_objective_no_target_deficit(subproblem)
                    state_in = get_next_state(subproblem, state_params_out[j], state_in, state_out)
                end
            end
            objective /= num_train_per_batch
            eval_loss /= num_train_per_batch
            return objective
        end
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end

function make_single_network(models::Vector{F}, number_of_states::Int) where {F}
    size_m = length(models)
    return Parallel(vcat, [Chain(
        x -> x[1:number_of_states * (i + 1)],
        models[i]
    ) for i in 1:size_m]...)
end

function train_multistage(models::Vector, initial_state, subproblems, state_params_in, state_params_out, uncertainty_sampler; 
    num_batches=100, num_train_per_batch=32, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
    adjust_hyperparameters=(iter, opt_state, num_train_per_batch) -> num_train_per_batch,
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end,
    get_objective_no_target_deficit=get_objective_no_target_deficit
)
    num_states = length(initial_state)
    model = make_single_network(models, num_states)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        num_train_per_batch = adjust_hyperparameters(iter, opt_state, num_train_per_batch)
        # Sample uncertainties
        uncertainty_samples = [sample(uncertainty_sampler) for _ in 1:num_train_per_batch]
        uncertainty_samples_vecs = [[collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)] for uncertainty_sample in uncertainty_samples]
        uncertainty_samples_vec = [vcat(initial_state, uncertainty_samples_vecs[s]...) for s in 1:num_train_per_batch]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        eval_loss = 0.0
        objective = 0.0
        grads = Flux.gradient(model) do m
            for s in 1:num_train_per_batch
                states = m(uncertainty_samples_vec[s])
                state_in = initial_state
                for (j, subproblem) in enumerate(subproblems)
                    state_out = ensure_feasibility(states[(j - 1) * num_states + 1:j * num_states], state_in, uncertainty_samples_vecs[s][j])
                    objective += simulate_stage(subproblem, state_params_in[j], state_params_out[j], uncertainty_samples[s][j], state_in, state_out)
                    eval_loss += get_objective_no_target_deficit(subproblem)
                    state_in = get_next_state(subproblem, state_params_out[j], state_in, state_out)
                end
            end
            objective /= num_train_per_batch
            eval_loss /= num_train_per_batch
            return objective
        end
        record_loss(iter, model, eval_loss, "metrics/loss") && break
        record_loss(iter, model, objective, "metrics/training_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end
