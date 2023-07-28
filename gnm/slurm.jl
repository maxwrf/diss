using Printf
using BenchmarkTools: @time

include("spike_train.jl")
include("gnm_utils.jl")

function generate_inputs(
    file_name::String,
    n_runs::Int,
    d_set_id::String,
    mea_id::String,
    dt::Float64
)
    # prepare the parameter space
    param_space = generate_param_space(n_runs)

    # generate the spike train
    spike_train = Spike_Train(
        file_name,
        mea_id,
        dt
    )

    m = sum(spike_train.A_Y) / 2
    m_max = (size(spike_train.A_Y, 1) * (size(spike_train.A_Y, 1) - 1)) / 2

    # if with the current dt and corr cutoff there are no connections, skip
    if m == 0
        println(file_name, " no connections, skipping.")
        return
    end

    println(file_name, " ", m, "/", m_max, "(", round(m / m_max * 100, digits=1), "%) connections.")

    # retrieve the recording number from the file name
    base, num_ext = split(file_name, '_')
    recording_num = parse(Int64, (split(num_ext, ".h5")[1]))

    # for every recording prepare thriteen files for each model
    for (model_id, model_name) in MODELS
        out_file = base * "_" * @sprintf("%05d", ((recording_num - 1) * length(MODELS)) + model_id) * ".dat"
        file = h5open(out_file, "w")

        # write the data
        write(file, "A_Y", spike_train.A_Y)
        write(file, "A_init", spike_train.A_init)
        write(file, "D", spike_train.D)
        write(file, "param_space", param_space)

        # write the meta data
        meta_group = create_group(file, "meta")
        attributes(meta_group)["org_file_name"] = spike_train.org_file_name
        attributes(meta_group)["data_set_id"] = d_set_id
        attributes(meta_group)["data_set_name"] = config["data_sets"][d_set_id]
        attributes(meta_group)["group_id"] = spike_train.group_id
        attributes(meta_group)["model_id"] = model_id
        attributes(meta_group)["model_name"] = model_name

        close(file)
    end

    println("Prepared ", file_name, "for network generation.")
end

function combine_res_files(in_dir::String)
    res_files = filter(name -> endswith(name, ".res"), readdir(in_dir))
    res_files = map(name -> joinpath(in_dir, name), res_files)

    K_all = Vector{Matrix{Float64}}() #  each K is param combs x 4 evals
    group_ids = Vector{String}()
    model_ids = Vector{Int}()
    param_space = nothing
    d_set_id = nothing
    data_set_name = nothing

    for (i_res_files, res_file) in enumerate(res_files)
        file = h5open(res_file, "r")
        K = read(file, "K")
        push!(K_all, K)

        if i_res_files == 1
            param_space = read(file, "param_space")
            meta_group = file["meta"]
            d_set_id = read_attribute(meta_group, "data_set_id")
            data_set_name = read_attribute(meta_group, "data_set_name")
        end

        # read meta data
        meta_group = file["meta"]
        group_id = read_attribute(meta_group, "group_id")
        model_id = read_attribute(meta_group, "model_id")
        push!(group_ids, group_id)
        push!(model_ids, model_id)

        close(file)
    end

    println("Read ", length(res_files), " result files.")

    # write the combined results
    unique_group_ids = unique(group_ids)
    for group_id in unique_group_ids
        # one file for every group with meta data in common & params in common
        file = h5open(joinpath(in_dir, "group_" * string(group_id) * ".h5"), "w")
        meta_group = create_group(file, "meta")
        attributes(meta_group)["data_set_id"] = d_set_id
        attributes(meta_group)["data_set_name"] = data_set_name
        attributes(meta_group)["group_id"] = group_id
        write(file, "param_space", param_space)

        # one dataset for every model in the results group
        result_group = create_group(file, "results")
        for (model_id, _) in MODELS
            indices = findall((group_ids .== group_id) .& (model_ids .== model_id))
            if length(indices) == 0
                println("Model ", model_id, " not found in group ", group_id, ".")
                continue
            end

            # need to prepare the size of the vector of matrices
            # number of samples x number of parameters x number of eval functions
            K_group_model = Array{Float64}(undef, length(indices), size(param_space, 1), 4)
            for (i, idx) in enumerate(indices)
                K_group_model[i, :, :] = K_all[idx]
            end
            write(result_group, string(model_id), K_group_model)
        end
        close(file)
    end

    println("Wrote ", length(unique_group_ids), " group result files.")
end
