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

function simulate_stage(subproblem::JuMP.Model, state_param_in::Vector{VariableRef}, state_param_out::Vector{VariableRef}, uncertainty::Dict{VariableRef, T}, state_in::Vector{Z}, state_out::Vector{V}
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
    for (i, state_var) in enumerate(state_param_out)
        MOI.set(subproblem, POI.ParameterValue(), state_var, state_out[i])
    end

    # Solve subproblem
    optimize!(subproblem)

    # Return objective value
    return JuMP.objective_value(subproblem)
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{VariableRef}},
    states::Vector{Vector{Float64}},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    )
    
    # Loop over stages
    objective_value = 0.0
    for stage in 1:length(subproblems)
        state_in = states[stage]
        state_out = states[stage + 1]
        subproblem = subproblems[stage]
        state_param_in = state_params_in[stage]
        state_param_out = state_params_out[stage]
        uncertainty = uncertainties[stage]
        objective_value += simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    end
    
    # Return final objective value
    return objective_value
end

function simulate_multistage(
    subproblems::Vector{JuMP.Model},
    state_params_in::Vector{Vector{VariableRef}},
    state_params_out::Vector{Vector{VariableRef}},
    initial_state::Vector{Float64},
    uncertainties::Vector{Dict{VariableRef, Float64}},
    decision_rules;
    ensure_feasibility=(x_out, x_in, uncertainty) -> x
)
    states = simulate_states(initial_state, uncertainties, decision_rules, ensure_feasibility=ensure_feasibility)
    return simulate_multistage(subproblems, state_params_in, state_params_out, states, uncertainties)
end

pdual(v::VariableRef) = MOI.get(JuMP.owner_model(v), POI.ParameterDual(), v)
pdual(vs::Vector{VariableRef}) = [pdual(v) for v in vs]

# Define rrule of simulate_multistage
function rrule(::typeof(simulate_stage), subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    y = simulate_stage(subproblem, state_param_in, state_param_out, uncertainty, state_in, state_out)
    function _pullback(Δy)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), pdual.(state_param_in) * Δy, pdual.(state_param_out) * Δy)
    end
    return y, _pullback
end

function sample(uncertainty_samples::Dict{VariableRef, Vector{Float64}})
    return Dict((k => v[rand(1:end)]) for (k, v) in uncertainty_samples)
end

sample(uncertainty_samples::Vector{Dict{VariableRef, Vector{Float64}}}) = [sample(uncertainty_samples[t]) for t in 1:length(uncertainty_samples)]

function train_multistage(model, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; 
    num_train_samples=100, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
    record_loss=(iter, x) -> println("Iter: $iter, Loss: $x")
)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_train_samples
        Flux.reset!(model)
        # Sample uncertainties
        uncertainty_sample = sample(uncertainty_samples)
        uncertainty_sample_vec = [collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)]

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        training_loss = 0.0
        grads = Flux.gradient(model) do m
            objective = 0.0
            state_in = initial_state
            for (j, subproblem) in enumerate(subproblems)
                state_out = m(uncertainty_sample_vec[j])
                state_out = ensure_feasibility(state_out, state_in, uncertainty_sample_vec[j])
                objective += simulate_stage(subproblem, state_params_in[j], state_params_out[j], uncertainty_sample[j], state_in, state_out)
                state_in = state_out
            end
            training_loss += objective
            return objective
        end
        record_loss(iter, training_loss)

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

function train_multistage(models::Vector, initial_state, subproblems, state_params_in, state_params_out, uncertainty_samples; 
    num_train_samples=100, optimizer=Flux.Adam(0.01), ensure_feasibility=(x_out, x_in, uncertainty) -> x_out,
    record_loss=(iter, x) -> println("Iter: $iter, Loss: $x")
)
    num_states = length(initial_state)
    model = make_single_network(models, num_states)
    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_train_samples
        # Sample uncertainties
        uncertainty_sample = sample(uncertainty_samples)
        uncertainty_sample_vecs = [collect(values(uncertainty_sample[j])) for j in 1:length(uncertainty_sample)]
        uncertainty_sample_vec = vcat(initial_state, uncertainty_sample_vecs...)

        # Calculate the gradient of the objective
        # with respect to the parameters within the model:
        training_loss = 0.0
        grads = Flux.gradient(model) do m
            objective = 0.0
            states = m(uncertainty_sample_vec)
            for (j, subproblem) in enumerate(subproblems)
                if j == 1
                    state_in = initial_state
                else
                    state_in = ensure_feasibility(states[(j - 2) * num_states + 1:(j-1) * num_states], states[(j - 1) * num_states + 1:j * num_states], uncertainty_sample_vecs[j - 1])
                end
                state_out = ensure_feasibility(states[(j - 1) * num_states + 1:j * num_states], state_in, uncertainty_sample_vecs[j])
                objective += simulate_stage(subproblem, state_params_in[j], state_params_out[j], uncertainty_sample[j], state_in, state_out)
            end
            training_loss += objective
            return objective
        end
        record_loss(iter, training_loss)

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
    
    return model
end
