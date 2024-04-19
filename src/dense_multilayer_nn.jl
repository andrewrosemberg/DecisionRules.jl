identity(x) = x

function dense_multilayer_nn(num_inputs::Int, num_outputs::Int, layers::Vector{Int}; activation=Flux.relu, dense=Dense)
    if length(layers) == 0
        if dense == LSTM
            return dense(num_inputs, num_outputs)
        end
        return dense(num_inputs, num_outputs, activation)
    end
    midlayers = []
    for i in 1:length(layers) - 1
        if dense == LSTM
            push!(midlayers, dense(layers[i], layers[i + 1]))
        else
            push!(midlayers, dense(layers[i], layers[i + 1], activation))
        end
    end
    first_layer = if dense == LSTM
        dense(num_inputs, layers[1])
    else
        dense(num_inputs, layers[1], activation)
    end
    model = Chain(first_layer, midlayers..., dense(layers[end], num_outputs))
    return model
end

function dense_multilayer_nn(num_models::Int, num_inputs::Int, num_outputs::Int, layers::Vector{Int}; activation=Flux.relu, dense=Dense)
    if num_models == 1
        return dense_multilayer_nn(num_inputs, num_outputs, layers, activation=activation, dense=dense)
    else
        return [dense_multilayer_nn(num_inputs * (i + 1), num_outputs, layers, activation=activation, dense=dense) for i in 1:num_models]
    end
end