using JLD2
using JSON
using Arrow
using Random
using StatsBase
using DataFrames
using ProgressMeter

using Flux
using ParameterSchedulers
using CUDA

rng = MersenneTwister(42)

# load data
train_ratio = 0.8
EPOCHS = 1000
input_data = DataFrame(Arrow.Table("data/bolivia_ACP_96_100000_input.arrow"))
output_data = DataFrame(Arrow.Table("data/bolivia_ACP_96_100000_output.arrow"))
bounds_data = JSON.parsefile("data/bolivia_ACP_96_100000_bounds.json")
obj_bounds = extrema(output_data[!, :objective])
bounds_data["objective"] = (obj_bounds[1] * 0.9, obj_bounds[2] * 1.1)

# filter out unnecessary columns
constant_input_columns = [(Symbol(c), input_data[1, c]) for c in names(input_data) if all(input_data[!, c] .== input_data[1, c])]
constant_output_columns = [(Symbol(c), output_data[1, c]) for c in names(output_data) if all(output_data[!, c] .== output_data[1, c])]
input_filter = Not([[:id]; first.(constant_input_columns)])
output_filter = Not([
    # select just primal variables
    [:id, :time, :status, :primal_status, :dual_status];
    [Symbol(n) for n in names(output_data) if occursin("dual_", n)];
    first.(constant_output_columns)
])
@info "Removing constant columns \nInput: " * join(constant_input_columns, ", ") * "\nOutput: " * join(constant_output_columns, ", ")

input_features = Matrix{Float32}(input_data[!, input_filter])'
output_variables = Matrix{Float32}(output_data[!, output_filter])'
input_bounds::Vector{Tuple{Float32, Float32}} = [(bounds_data[n][1], bounds_data[n][2]) for n in names(input_data[!, input_filter])]
output_bounds::Vector{Tuple{Float32, Float32}} = [(bounds_data[n][1], bounds_data[n][2]) for n in names(output_data[!, output_filter])]

# split train/test
n_train = Int(floor(size(input_features, 2) * train_ratio))
n_test = size(input_features, 2) - n_train
rand_perm = randperm(rng, size(input_features, 2))

train_indices, test_indices = rand_perm[1:n_train], rand_perm[n_train+1:end]
train_input_un, test_input_un = input_features[:, train_indices], input_features[:, test_indices]
train_output_un, test_output_un = output_variables[:, train_indices], output_variables[:, test_indices]

# normalize input/output to [0, 1] using bounds
normalize(x, bounds) = (x .- bounds[1]) ./ (bounds[2] - bounds[1])
scale(x, bounds) = x .* (bounds[2] - bounds[1]) .+ bounds[1]
train_input = normalize.(train_input_un, input_bounds)
test_input = normalize.(test_input_un, input_bounds)
train_output = normalize.(train_output_un, output_bounds)
test_output = normalize.(test_output_un, output_bounds)

@assert scale.(train_input, input_bounds) ≈ train_input_un
@assert scale.(test_input, input_bounds) ≈ test_input_un
@assert scale.(train_output, output_bounds) ≈ train_output_un
@assert scale.(test_output, output_bounds) ≈ test_output_un

# skip connect for x_hat
xhat_idx = findall(n -> occursin("_out", n), names(input_data[!, input_filter]))
@assert length(xhat_idx) == size(output_variables, 1) - 1
@assert "objective" == names(output_data[!, output_filter])[end]
function xhat_skipconnect(mx, x)
    # add zero row for objective (luckily it is last)
    xhat = vcat(x[xhat_idx, :], zeros(1, size(x, 2)))
    return xhat .+ mx
end

hidden_dim = 32
model = Chain(
    # SkipConnection(
        # Chain(
            Dense(size(input_features, 1), hidden_dim, softplus),
            # SkipConnection(
            #     Chain(
            #         SkipConnection(
                        Dense(hidden_dim, hidden_dim, softplus),
                    # +),
                    # SkipConnection(
                        Dense(hidden_dim, hidden_dim, softplus),
                    # +),
                    # SkipConnection(
                        Dense(hidden_dim, hidden_dim, softplus),
            #         +),
            #     ),
            #     +
            # ),
            Dense(hidden_dim, size(output_variables, 1)),
    #     ), xhat_skipconnect
    # ),
)

println(); Flux._big_show(stdout, model); println()

# early stopping callbacks
overfit_count = 0
best_test_loss = Inf32
function overfit_early_stop(train_loss, test_loss)
    if mean(test_loss) > mean(train_loss) * 1.1
        global overfit_count += 1
    else
        global overfit_count = 0
    end
end
noimprove_count = 0
function noimprove_early_stop(test_loss)
    if mean(test_loss) > mean(best_test_loss)
        global noimprove_count += 1
    else
        global noimprove_count = 0
    end
end

# optimizer/lr scheduler
optimizer = Flux.Adam()
optimizer_st = Flux.setup(optimizer, model)
lr_scheduler = Exp(1e-3, 1 - (2/EPOCHS))

# move to gpu
model = model |> gpu
optimizer = optimizer |> gpu
train_input = train_input |> gpu
train_output = train_output |> gpu
test_input = test_input |> gpu
test_output = test_output |> gpu

# make dataloader
data = Flux.DataLoader((train_input, train_output), batchsize=8, shuffle=true) |> gpu

# define loss and validation callbacks
# predict(x) = sigmoid(model(x))
function predict(x)
    y = model(x)
    softplus(y) .- softplus(y .- 1.0f32)
end
loss(x, y) = Flux.mse(predict(x), y)
mae(x, y) = mean(abs.((x .- y)), dims=2)
train_loss() = mae(predict(train_input), train_output) |> cpu
test_loss() = mae(predict(test_input), test_output) |> cpu

# run training
p = Progress(EPOCHS, showspeed=true, desc="Training...")
epochs = 1:EPOCHS
for (eta, e) in zip(lr_scheduler, epochs)
    # lr adjustment
    Flux.Optimisers.adjust!(optimizer_st, eta)

    # training
    start_time = time()
    Flux.train!(loss, Flux.params(model), data, optimizer)
    end_time = time()


    # validation/logging
    train_loss_ = train_loss()
    test_loss_ = test_loss()
    mean_train_loss = mean(train_loss_)
    mean_test_loss = mean(test_loss_)
    overfit_early_stop(mean_train_loss, mean_test_loss)
    noimprove_early_stop(mean_test_loss)
    global best_test_loss = min(best_test_loss, mean_test_loss)

    next!(p, showvalues=[
        (:epoch, e),
        (:lr, eta),
        (:time, end_time - start_time),
        (:train_loss, train_loss_),
        (:test_loss, test_loss_),
        (:mean_train_loss, mean_train_loss),
        (:mean_test_loss, mean_test_loss),
        (:best_test_loss, best_test_loss),
        (:overfit, overfit_count),
        (:noimprove, noimprove_count)
    ])

    overfit_count > Inf && (@warn "Early stopping due to overfitting"; break)
    noimprove_count > Inf && (@warn "Early stopping due to no improvement"; break)
end
finish!(p, keep=true)
@info "Training complete" train_loss() test_loss()

# save model (load with Flux.loadmodel!(model, JLD2.load("proxy.jld2", "state")))
@info "Saving model"
jldsave("proxy.jld2"; state=model |> cpu |> Flux.state)

@info "Done"