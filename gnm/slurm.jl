include("spike_set.jl")
include("gnm_utils.jl")

function generate_inputs(
    in_dir::String,
    out_dir::String,
    n_samples::Int,
    n_runs::Int,
    d_set_id::Int,
    mea_id::Int,
    dt::Float64,
    corr_cutoff::Float64
)
    spike_set = Spike_Set(in_dir, mea_id, d_set_id, n_samples, dt, corr_cutoff)
    param_space = generate_param_space(n_runs)

    # check the output directory exists
    if !isdir(out_dir)
        mkdir(out_dir)
    end

    for (i_spike_train, spike_train) in enumerate(spike_set.spike_trains)
        for (model_idx, model_name) in MODELS
            out_file = out_dir * "/sample_" * string(i_spike_train) * "_model_" * string(model_idx) * ".dat"
            file = h5open(out_file, "w")

            # write the data
            write(file, "A_Y", spike_train.A_Y)
            write(file, "A_init", spike_train.A_init)
            write(file, "D", spike_set.D)
            write(file, "param_space", param_space)

            # write the meta data
            meta_group = create_group(file, "meta")
            attributes(meta_group)["data_set_id"] = d_set_id
            attributes(meta_group)["data_set_name"] = DATA_SETS[d_set_id]
            attributes(meta_group)["group_id"] = spike_train.group_id
            attributes(meta_group)["model_idx"] = model_idx
            attributes(meta_group)["model_name"] = model_name

            close(file)
        end
    end
end

