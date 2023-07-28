using Random: randperm


function jitter_spikes_fast(
    spikes::Vector{Float64},
    spike_counts::Vector{Int32},
    stdev::Float64
)
    """
    Important to match the stdev unit and the unit of the spikes

    Why would you need rand perm?
    Why would you round?

    I am not removing duplicates, which are soo unlikely to occur
    Also what would be the implications?
    """
    n_electrodes = length(spike_counts)
    st_cumsum = [0; cumsum(spike_counts)]
    max_rec_time = maximum(spikes)
    min_rec_time = minimum(spikes)

    jittered_spikes = spikes .+ (stdev * randn(length(spikes)))

    jittered_spikes[jittered_spikes.<min_rec_time] = spikes[jittered_spikes.<min_rec_time]
    jittered_spikes[jittered_spikes.>max_rec_time] = spikes[jittered_spikes.>max_rec_time]


    # Jitter the data, for every electrode
    for i_channel in 1:n_electrodes

        # duplicate error
        #dup_idxs = findall(spike -> count(spike .== st_jittered) > 1, st_jittered)
        #st_jittered[dup_idxs] = st[dup_idxs]

        # replace
        jittered_spikes[(st_cumsum[i_channel]+1):(st_cumsum[i_channel+1])] = sort(jittered_spikes[(st_cumsum[i_channel]+1):(st_cumsum[i_channel+1])])
    end

    return jittered_spikes
end



# using BenchmarkTools: @btime
# file_path = "/store/DAMTPEGLEN/mw894/data/Demas2006/sample_1/Demas2006_p21nob1.h5"
# file = h5open(file_path, "r")
# spikes = read(file, "spikes")
# spike_counts = read(file, "sCount")
# close(file)
# @btime jitter_spikes_fast(spikes, spike_counts, 0.01)