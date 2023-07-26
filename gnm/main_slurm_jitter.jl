include("jitter.jl")
include("sttc.jl")

using HDF5
using .GNM_Mod

function main(test_path::Union{String,Nothing}=nothing)
    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end


    # read data
    file = h5open(file_path, "r")
    spikes = read(file, "spikes")
    spike_counts = read(file, "sCount")
    close(file)
    recording_time = [minimum(spikes), maximum(spikes)]

    # main loop
    jittered = jitter_spikes_fast(spikes, spike_counts, 0.01)
    jitter_spikes(spikes, spike_counts, 0.01)

    jittered_sttc = zeros(num_permutations, size(sttc)...)
    for i in 1:num_permutations
        jittered_spikes = jitter_spikes_fast(spikes, spike_counts, dt_jitter)
        jittered_sttc[i, :, :] = sttc_tiling(dt, recording_time, jittered_spikes, spike_counts)
    end



    # save results
    res_file_path = replace(file_path, r"\.h5$" => ".jitter")
    file = h5open(res_file_path, "w")
    write(file, "K", model.K)
    write(file, "param_space", param_space)
    close(file)
end


main("/store/DAMTPEGLEN/mw894/slurm/Charlesworth2015/sample_00001.dat")