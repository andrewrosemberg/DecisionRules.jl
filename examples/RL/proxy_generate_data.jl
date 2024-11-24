using Base.Threads
using ProgressMeter

using JuMP
using Ipopt

using Random
using Distributions

using Arrow
using DecisionRules
using LearningToOptimize

include("../HydroPowerModels/load_hydropowermodels.jl")


if length(ARGS) == 0
    case_name, formulation, num_stages, num_samples = "bolivia", "ACPPowerModel", 96, 100_000
elseif length(ARGS) == 4
    case_name, formulation, num_stages, num_samples = ARGS
else
    println("Usage: julia proxy_generate_data.jl <case_name> <formulation> <num_stages> <num_samples>")
    println("  Got: $(ARGS)")
    exit(1)
end

MaybeUniform(a, b) = (a == b) ? Dirac(a) : Uniform(a, b)

function make_problem_iterator(case_name, formulation, num_stages, num_samples; rng=nothing)
    # load data with hydropowermodels
    (
        subproblems,
        state_params_in,
        state_params_out,
        uncertainty_samples,
        initial_state,
        max_volume
    ) = build_hydropowermodels(
        joinpath("../HydroPowerModels", case_name),
        formulation * ".mof.json";
        num_stages=num_stages,
        param_type=:Var # NOTE: change to param for L2O
    )

    # get first jump model/var refs for L2O to use
    subproblem = subproblems[1]
    state_param_in = state_params_in[1]
    state_param_out = state_params_out[1]
    
    set_optimizer(subproblem, Ipopt.Optimizer)
    set_silent(subproblem)

    # extract sorted uncertainty variable refs.
    #   since each index is a dict, we sort by the string of the variable names.
    #   this is necessary to ensure we grab the correct uncertainties across problems
    str_uncertainty = [sort(collect(Dict((name(k), k) => v for (k, v) in sample))) for sample in uncertainty_samples]
    var_refs = [first.(str_sample) for str_sample in str_uncertainty]
    @assert all(first.(var_ref) == first.(var_refs[1]) for var_ref in var_refs)
    var_refs = last.(var_refs[1])

    # get bounds for each uncertainty variable across all samples
    bounds = extrema.(eachcol(vcat(stack.([last.(str_sample) for str_sample in str_uncertainty])...)))

    # convert to uniform (or dirac) distribution for each uncertainty variable
    dists = [MaybeUniform(bound...) for bound in bounds]

    # sample uncertainty, state_in, and state_out_target
    if isnothing(rng)
        rng = MersenneTwister(42)
    end

    uncertainty = rand.(rng, dists, num_samples)
    state_in = rand.(rng, [MaybeUniform(0.0, max_volume[i]) for i in 1:length(state_param_in)], num_samples)
    state_out_target = rand.(rng, [MaybeUniform(0.0, max_volume[i]) for i in 1:length(state_param_out)], num_samples)

    # make dictionary mapping from variable to vector of values for L2O ProblemIterator
    #  don't need to sort state_param_in/out since they are vectors already
    uncertainty_map = Dict(var_refs[i] => uncertainty[i] for i in 1:length(var_refs))
    state_in_map = Dict(state_param_in[i] => state_in[i] for i in 1:length(state_param_in))
    state_out_target_map = Dict(first(state_param_out[i]) => state_out_target[i] for i in 1:length(state_param_out))

    combined_map::Dict{VariableRef, Vector{Float32}} = merge(uncertainty_map, state_in_map, state_out_target_map)

    # extract the fixing constraints, since we may want those duals.
    constr_refs::Vector{ConstraintRef} = []
    for c in all_constraints(subproblem, include_variable_in_set_constraints=false)
        cof = constraint_object(c).func
        for i in 1:length(state_param_out)
            (typeof(cof) <: GenericAffExpr) || continue # only affine constraints
            # the only constraint that in includes the parameter should be the fixing one
            if (first(state_param_out[i]) in keys(cof.terms))
                set_name(c, "_fixing_$(name(last(state_param_out[i])))")
                push!(constr_refs, c)
            end
        end
    end
    # should be one per state_param_out
    @assert length(constr_refs) == length(state_param_out)

    return (
        ProblemIterator(combined_map; param_type=LearningToOptimize.JuMPParameterType),
        last.(state_param_out),
        constr_refs,
        Dict([
            [name(v) => b for (b, v) in zip(bounds, var_refs)];
            [name(state_param_in[i]) => (0.0, max_volume[i]) for i in 1:length(state_param_in)];
            [name(state_param_out[i][1]) => (0.0, max_volume[i]) for i in 1:length(state_param_out)];
            [name(state_param_out[i][2]) => (0.0, max_volume[i]) for i in 1:length(state_param_out)]
        ])
    )
end

function threaded_solve_batch(problem_iterator, recorder)
    successfull_solves = Threads.Atomic{Int}(0.0)
    @showprogress desc="Solving..." Threads.@threads for idx in 1:length(problem_iterator.ids)
        _success_bool, _ = LearningToOptimize.solve_and_record(problem_iterator, recorder, idx)
        Threads.atomic_add!(successfull_solves, Int(_success_bool))
        # ignoring early_stop
    end
    @info "Recorded $(successfull_solves[]) of $(length(problem_iterator.ids)) problems"
    return successfull_solves[]
end


function generate_data(case_name, formulation, num_stages, num_samples)
    start = time()
    @info "Generating $(case_name) dataset with $(num_samples) samples with $(num_stages) $(formulation) stages"

    @info "Making problem iterator"
    problem_iterator, primal_variables, dual_variables, bounds = make_problem_iterator(
        case_name, formulation, num_stages, num_samples
    )

    output_dir = "data/"
    mkpath(output_dir); !isfile(output_dir * ".gitignore") && open(output_dir * ".gitignore", "w") do io println(io, "*"); end
    
    output_file_name = (
        case_name * "_" *
        split(formulation, "PowerModel")[1] * "_" *
        string(num_stages) * "_" *
        string(num_samples)
    )

    save(problem_iterator, output_dir * output_file_name * "_input", ArrowFile)

    recorder = Recorder{ArrowFile}(
        output_dir * output_file_name * "_output",
        primal_variables=primal_variables,
        dual_variables=dual_variables
    )

    @info "Solving batch"
    threaded_solve_batch(problem_iterator, recorder)

    @info "Merging outputs"
    file_outs = [
        file for file in readdir(output_dir; join=true)
            if (last(splitext(file)) == ".arrow") && (occursin("_output", file))
    ]
    # output_table = Arrow.Table(file_outs)
    # Arrow.write(
    #     output_dir * output_file_name * "_output.arrow",
    #     output_table,
    # )

    #merge in batches
    batchsize = 1024
    n_batches = Int(ceil(length(file_outs) / batchsize))
    batch_files = [output_dir * output_file_name * "_output_batch$(i).arrow" for i in 1:n_batches]
    Threads.@threads for (i, batch) in collect(enumerate(batch_files))
        batch_table = Arrow.Table(file_outs[((i-1)*batchsize+1):min(i*batchsize, length(file_outs))])
        Arrow.write(batch, batch_table)
        println("Wrote batch $(i) of $(n_batches)")
    end

    # level 2 batches
    batchsize2 = 4
    n_batches2 = Int(ceil(length(batch_files) / batchsize2))
    batch_files2 = [output_dir * output_file_name * "_output_2batch$(i).arrow" for i in 1:n_batches2]
    Threads.@threads for (i, batch) in collect(enumerate(batch_files2))
        batch_table = Arrow.Table(batch_files[((i-1)*batchsize2+1):min(i*batchsize2, length(batch_files))])
        Arrow.write(batch, batch_table)
        println("Wrote batch $(i) of $(n_batches2)")
    end

    # final merge
    final_table = Arrow.Table(batch_files2)
    Arrow.write(output_dir * output_file_name * "_output.arrow", final_table)

    @info "Deleting intermediate files"
    for file in batch_files2
        rm(file)
    end
    for file in batch_files
        rm(file)
    end
    for file in file_outs
        rm(file)
    end
    bounds_file = output_dir * output_file_name * "_bounds.json"
    open(bounds_file, "w") do io JSON.print(io, bounds) end

    @info "Generated $(num_samples) samples of $(case_name) with $(num_stages) $(formulation) stages in $(time() - start) seconds"
end

generate_data(case_name, formulation, num_stages, num_samples)