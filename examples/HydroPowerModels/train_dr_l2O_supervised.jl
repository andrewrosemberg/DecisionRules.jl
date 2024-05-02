using Statistics
using Random
using Flux
using Arrow
using JSON
using DataFrames
using DecisionRules
using CUDA

using Wandb, Dates, Logging
using JLD2

case_name = "case3"
formulation = "ACPPowerModel"
num_stages = 48
batch_size = 32
optimizer = Flux.Adam(0.01)

save_file = "supervised-$(case_name)-$(formulation)-h$(num_stages)-$(now())"

HydroPowerModels_dir = dirname(@__FILE__)
case_dir = joinpath(HydroPowerModels_dir, case_name)
model_dir = joinpath(case_dir, formulation, "models")
output_dir = joinpath(case_dir, formulation, "output")

hydro_file = JSON.parsefile(joinpath(case_dir, "hydro.json"))

num_hydro = length(hydro_file["Hydrogenerators"])
stage_hours = hydro_file["stage_hours"]
volume_to_mw(volume, stage_hours; k = 0.0036) = volume / (k * stage_hours)

input_files = [file for file in readdir(case_dir; join=true) if (
        occursin(case_name, file) && occursin(formulation, file) && occursin("arrow", file) && occursin("input", file)
    )
]

output_files = [file for file in readdir(output_dir; join=true) if (
        occursin(case_name, file) && occursin(formulation, file) && occursin("arrow", file) && occursin("output", file)
    )
]

input_table = deepcopy(DataFrame(Arrow.Table(input_files)))
output_table = DataFrame(Arrow.Table(output_files))

for i=1:num_hydro
    input_table[:, Symbol("_inflow[$i]#1")] = input_table[:, Symbol("_inflow[$i]#1")] .+ volume_to_mw.(input_table[:, Symbol("_reservoir[$i]_in#1")],  stage_hours)
end

input_names = [[Symbol("_inflow[$i]#$t") for i=1:num_hydro] for t = 1:num_stages]
output_names = [[Symbol("reservoir[$i]_out#$t") for i=1:num_hydro] for t = 1:num_stages]

data_table = innerjoin(input_table, output_table; on=:id)
data_table[!, vcat(input_names...)] = Float32.(data_table[:, vcat(input_names...)])
data_table[!, vcat(output_names...)] = Float32.(data_table[:, vcat(output_names...)])

data_table = gpu(data_table)

model = gpu(Chain(Dense(num_hydro, 8, relu), LSTM(8, 8), Dense(8, num_hydro)))

function train_test(model, data_table, num_hydro, num_stages, input_names, output_names; loss=Flux.mse, batch_size=32,  optimizer = Flux.Adam(0.01),
    record_loss=(iter, model, loss, tag) -> begin println("tag: $tag, Iter: $iter, Loss: $loss")
        return false
    end
    )
    num_samples = length(data_table.id)
    num_batches = ceil(Int, num_samples / batch_size)

    # Initialise the optimiser for this model:
    opt_state = Flux.setup(optimizer, model)

    for iter in 1:num_batches
        iter_data_table = data_table[(iter - 1)*batch_size+1:min(iter*batch_size, num_samples), :]
        objective = 0.0
        grads = Flux.gradient(model) do m
            for s in 1:length(iter_data_table.id)
                Flux.reset!(m)
                target_states = hcat([m([iter_data_table[s,input_names[t][i]] for i=1:num_hydro]) for t=1:num_stages]...)
                optimal_states = hcat([[iter_data_table[s,output_names[t][i]] for i=1:num_hydro] for t=1:num_stages]...)
                objective += loss(target_states, optimal_states)
            end
            objective /= batch_size
            return objective
        end
        record_loss(iter, model, objective, "metrics/batch_loss") && break

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end
end

model_path = joinpath(model_dir, save_file * ".jld2")

save_control = SaveBest(100, model_path, 0.003)

lg = WandbLogger(
    project = "HydroPowerModels",
    name = save_file,
    config = Dict(
        "Supervised" => "Yes",
        "optimizer" => "Adam"
    )
)

train_test(model, data_table, num_hydro, num_stages, input_names, output_names;
    record_loss= (iter, model, loss, tag) -> begin
        save_control(iter, model, loss)
        return record_loss(iter, model, loss, tag)
    end
)
