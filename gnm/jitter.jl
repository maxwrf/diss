using Random: randperm

function jitter_spikes(
    spikes::Vector{Float64},
    spike_counts::Vector{Int32},
    stdev::Float64
)
    """
    Important to match the stdev unit and the unit of the spikes
    """

    n_electrodes = length(spike_counts)
    st_cumsum = [0; cumsum(spike_counts)]

    jittered_spikes = copy(spikes)


    max_rec_time = maximum(spikes)


    # Jitter the data, for every electrode
    for i_channel in 1:n_electrodes

        # Get the number of spikes
        nSpikes = st_cumsum[i_channel+1] - st_cumsum[i_channel]

        # get the spikes and create a copy for jittering
        st = spikes[(st_cumsum[i_channel]+1):(st_cumsum[i_channel+1])]
        st_jittered = copy(st)

        # Make a random ordering of which spikes will be jittered when
        jitterorder = randperm(nSpikes)

        for iSpike = 1:nSpikes
            # Jitter the spike
            st_jittered[jitterorder[iSpike]] = round(st_jittered[jitterorder[iSpike]] + stdev * randn())

            # Error check
            if st_jittered[jitterorder[iSpike]] < 1
                # Jittered off the front of the spike train
                st_jittered[jitterorder[iSpike]] = st[jitterorder[iSpike]]


            elseif st_jittered[jitterorder[iSpike]] > max_rec_time
                # Jittered off the end of the spike train
                st_jittered[jitterorder[iSpike]] = st[jitterorder[iSpike]]

            elseif length(unique(st_jittered)) < nSpikes
                # Jittered into another spike
                st_jittered[jitterorder[iSpike]] = st[jitterorder[iSpike]]
            end
        end

        sort!(st_jittered)
        jittered_spikes[(st_cumsum[i_channel]+1):(st_cumsum[i_channel+1])] = st_jittered
    end

    return jittered_spikes
end