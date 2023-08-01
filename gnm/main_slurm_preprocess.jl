using JSON
using Printf
using BenchmarkTools: @time

include("gnm_utils.jl")
include("spike_train.jl")

const config = JSON.parsefile("/home/mw894/diss/gnm/config.json")

function main(test_path::Union{String,Nothing}=nothing)
    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end

    println("File: ", file_path)
    println("Runs: ", config["params"]["n_runs"])

    # prepare the parameter space
    param_space = generate_param_space(config["params"]["n_runs"])

    # generate the spike train
    spike_train = Spike_Train(file_path)
    m = sum(spike_train.A_Y) / 2
    m_max = (size(spike_train.A_Y, 1) * (size(spike_train.A_Y, 1) - 1)) / 2

    # if with the current dt and corr cutoff there are no connections, skip
    if m == 0
        println(file_path, " no connections, skipping.")
        return
    end

    println(file_path, " ", m, "/", m_max, "(", round(m / m_max * 100, digits=1), "%) connections.")

    # retrieve the recording number from the file name
    base, num_ext = split(file_path, '_')
    recording_num = parse(Int64, (split(num_ext, ".h5")[1]))

    # for every recording prepare 13 files for each model
    for (model_id, model_name) in MODELS
        if config["params"]["subsample"]
            out_file = base * "_" * @sprintf("%05d", ((recording_num - 1) * length(MODELS)) + model_id) * ".subdat"
        else
            out_file = base * "_" * @sprintf("%05d", ((recording_num - 1) * length(MODELS)) + model_id) * ".dat"
        end

        file = h5open(out_file, "w")

        # write the data
        write(file, "A_Y", spike_train.A_Y)
        write(file, "A_init", spike_train.A_init)
        write(file, "D", spike_train.D)
        write(file, "param_space", param_space)
        write(file, "sttc", spike_train.sttc)

        # write the meta data
        meta_group = create_group(file, "meta")
        attributes(meta_group)["org_file_name"] = spike_train.org_file_name
        attributes(meta_group)["data_set_id"] = spike_train.dset_id
        attributes(meta_group)["data_set_name"] = config["data_sets"][string(spike_train.dset_id)]
        attributes(meta_group)["group_id"] = spike_train.group_id
        attributes(meta_group)["model_id"] = model_id
        attributes(meta_group)["model_name"] = model_name

        close(file)
    end

    println("Prepared ", file_path, "for network generation.")
end

#main("/store/DAMTPEGLEN/mw894/data/Demas2006/sample_00009.h5")
main()
