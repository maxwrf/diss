using HDF5
include("sttc.jl")

struct Spike_Train
    spikes::Vector{Float64}
    spike_counts::Vector{Float64}
    num_sample_electrodes::Int
    sample_electrode_names::Vector{String}
    group_id::String
    recording_time::Vector{Float64}
    sttc::Matrix{Float64} # active by active electrodes
    A_Y::Matrix{Float64} # all by all electrodes
    A_init::Matrix{Float64} # all by all electrodes
    m::Int # number of edges

    function Spike_Train(
        file_path::String,
        dset_type::Int,
        mea_type::Int,
        all_electrodes::Vector{Tuple{Int,Int}})

        # read the h5 data
        file = h5open(file_path, "r")
        spikes = read(file, "spikes")
        spike_counts = read(file, "sCount")
        recording_time = [minimum(spikes), maximum(spikes)]
        sample_electrode_names = read(file, "names")

        if (dset_type == 1)
            age = read(file, "meta/age")[1]
            region = read(file, "meta/region")[1]
            group_id = region * string(age)
        elseif (dset_type == 2)
            # TODO
        end
        close(file)

        # prepare the samples
        dt = 0.05
        sttc = sttc_tiling(dt, recording_time, spikes, spike_counts)
        sample_electrode_names, sttc, A_Y, A_init, m = prepare_sample(
            mea_type,
            sample_electrode_names,
            all_electrodes,
            sttc
        )

        # construct the sample
        new(
            spikes,
            spike_counts,
            length(spike_counts),
            sample_electrode_names,
            group_id,
            recording_time,
            sttc,
            A_Y,
            A_init,
            m
        )
    end
end

function prepare_sample(
    mea_type::Int,
    sample_electrode_names::Vector{String},
    all_electrodes::Vector{Tuple{Int,Int}},
    sttc::Matrix{Float64})

    # For the samples find the electrod names as tuples
    # Then identify the position of thes on the mea (needs all electrodes)
    # Sometimes electrodes are invalid and need to be removed
    removal_indices = []
    sample_electrodes = Vector{Tuple{Int,Int}}()
    if (mea_type == 1)
        for (i_active_electrode, sample_electrode_name) in enumerate(sample_electrode_names)
            if (sample_electrode_name[6] == 'a' || sample_electrode_name[6] == 'A')
                x = Int(sample_electrode_name[4]) - Int('0')
                y = Int(sample_electrode_name[5]) - Int('0')
                push!(sample_electrodes, (x, y))
            else
                push!(removal_indices, i_active_electrode)
            end
        end
    elseif (mea_type == 2) # TODO
    end

    # Remove the invalid electrodes
    splice!(sample_electrode_names, removal_indices)
    sttc = sttc[setdiff(1:size(sttc, 1), removal_indices),
        setdiff(1:size(sttc, 1), removal_indices)]

    # For the index of the electrodes in the vector of all electrodes
    sample_electrode_idx = findall(in(sample_electrodes), all_electrodes)

    # Fill the adjacency matrix and the initial adjacency matrix
    sttc_cutoff = 0.2
    A_Y = zeros(length(all_electrodes), length(all_electrodes))
    A_init = zeros(length(all_electrodes), length(all_electrodes))
    for i in 1:length(sample_electrodes)
        for j in (i+1):length(sample_electrodes)
            if sttc[i, j] > sttc_cutoff
                A_Y[sample_electrode_idx[i], sample_electrode_idx[j]] = 1
                A_Y[sample_electrode_idx[j], sample_electrode_idx[i]] = 1

                if rand() < 0.2
                    A_init[sample_electrode_idx[i], sample_electrode_idx[j]] = 1
                    A_init[sample_electrode_idx[j], sample_electrode_idx[i]] = 1
                end
            end
        end
    end

    m = sum(A_Y) / 2

    return sample_electrode_names, sttc, A_Y, A_init, m
end



# p = "/Users/maxwuerfek/code/diss/data/Charlesworth2015/TC190-DIV26_A.h5"
# x = Spike_Train(p, 1)
