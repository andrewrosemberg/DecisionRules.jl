function dense_multilayer_nn(num_inputs::Int, num_outputs::Int, layers::Vector{Int}; activation=Flux.relu, dense=Dense)
    if length(layers) == 0
        return dense(num_inputs, num_outputs, activation)
    end
    model = Chain(dense(num_inputs, layers[1], activation), [dense(layers[i], layers[i + 1], activation) for i in 1:length(layers) - 1]..., dense(layers[end], num_outputs))
    return model
end

function dense_multilayer_nn(num_models::Int, num_inputs::Int, num_outputs::Int, layers::Vector{Int}; activation=Flux.relu, dense=Dense)
    if num_models == 1
        return dense_multilayer_nn(num_inputs, num_outputs, layers, activation=activation, dense=dense)
    else
        return [dense_multilayer_nn(num_inputs * (i + 1), num_outputs, layers, activation=activation, dense=dense) for i in 1:num_models]
    end
end