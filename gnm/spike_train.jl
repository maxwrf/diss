using HDF5
using LinearAlgebra: triu, Symmetric, diagind
using StatsBase: sample
using BenchmarkTools: @time
using Distances: Euclidean, pairwise

include("sttc.jl")
include("jitter.jl")

mutable struct Spike_Train
    file_path::String
    dset_id::Int
    org_file_name::String
    group_id::String
    electrode_names::Vector{String}
    electrode_positions::Matrix{Float64}

    A_Y::Matrix{Float64}
    A_init::Matrix{Float64}
    D::Matrix{Float64}

    sttc::Matrix{Float64}

    function Spike_Train(file_path::String)
        # read the h5 data
        file = h5open(file_path, "r")
        spikes = read(file, "spikes")
        spike_counts = read(file, "sCount")
        recording_time = [minimum(spikes), maximum(spikes)]
        electrode_names = read(file, "names")
        electrode_positions = Matrix(read(file, "epos"))
        firing_rates = read(file, "/summary/frate")
        group_id = string(read(file, "meta/age")[1])
        org_file_name = read(file, "meta/org_file_name")
        dset_id = read(file, "meta/data_set_id")
        close(file)

        # subsample the spikes
        if config["params"]["subsample"]
            spikes, spike_counts, firing_rates, recording_time = subsample_strains(
                spikes,
                spike_counts,
                recording_time
            )
        end

        # filter the electrodes
        electrode_names, electrode_positions, spikes, spike_counts = filter_electrodes(
            dset_id,
            electrode_names,
            electrode_positions,
            firing_rates,
            spikes,
            spike_counts
        )

        # functional connectivity inference, i.e., adjacency matrix
        A_Y, sttc = functional_connectivity_inference(
            spikes,
            spike_counts,
            recording_time
        )
        A_Y[diagind(A_Y)] .= 0
        m = sum(A_Y) / 2

        # prepare the initalization matrix
        A_init = zeros(size(A_Y))
        edges = findall(==(1), triu(A_Y, 1))
        init_edges = sample(edges, Int(round(m * 0.2)); replace=false)
        A_init[init_edges] .= 1
        A_init = Symmetric(A_init, :U)

        # construct the distance matrix
        D = pairwise(Euclidean(), electrode_positions, dims=1)

        # construct the sample
        new(
            file_path,
            dset_id,
            org_file_name,
            group_id,
            electrode_names,
            electrode_positions,
            A_Y,
            A_init,
            D,
            sttc
        )
    end
end

function subsample_strains(spikes, spike_counts, recording_time)
    # compute the number of electrodes
    n_electrodes = length(spike_counts)

    # compute the new max recording time
    new_max_time = recording_time[end] * config["params"]["subsample_factor"]
    new_recording_time = Float64[recording_time[1], new_max_time]

    # adapt the spikes and spike counts, and firing rate
    st_cumsum = [0; cumsum(spike_counts)]

    new_spike_counts = Int32[]
    new_spikes = Float64[]
    new_firing_rates = Float64[]
    for i in 1:n_electrodes
        st = spikes[(st_cumsum[i]+1):(st_cumsum[i+1])]
        st = st[st.<new_max_time]
        push!(new_spike_counts, length(st))
        append!(new_spikes, st)
        push!(new_firing_rates, length(st) / new_max_time)
    end

    return new_spikes, new_spike_counts, new_firing_rates, new_recording_time
end

function functional_connectivity_inference(
    spikes::Vector{Float64},
    spike_counts::Vector{Int32},
    recording_time::Vector{Float64}
)
    # compute experimental sttc
    sttc = sttc_tiling(config["params"]["dt"], recording_time, spikes, spike_counts)

    # prepare permutations
    perm_test_counts = zeros(size(sttc))
    for i in 1:config["params"]["num_perms"]
        jittered_spikes = jitter_spikes_fast(spikes, spike_counts, config["params"]["dt_jitter"])
        jittered_sttc = sttc_tiling(config["params"]["dt"], recording_time, jittered_spikes, spike_counts)
        perm_test_counts .+= (jittered_sttc .>= sttc)
    end

    # compute p_value as in permutation test
    p_vals = perm_test_counts ./ config["params"]["num_perms"]
    functional_connects = Int.(p_vals .<= config["params"]["p_value"])

    return functional_connects, sttc
end


function filter_electrodes(
    dset_id::Int,
    electrode_names::Vector{String},
    electrode_positions::Matrix{Float64},
    firing_rates::Vector{Float64},
    spikes::Vector{Float64},
    spike_counts::Vector{Int32}
)
    """
    Electrodes are invalid if:
    1. Less than a certain Hz number
    2. Two neurons on the same electrode, because then the distance is zero
    """
    removal_indices = []

    # For different dataset different indications for a second neuron at the electrode
    if ((dset_id == 1) || (dset_id == 2))
        for (i_active_electrode, electrode_name) in enumerate(electrode_names)
            if !((electrode_name[end] == '0') && (firing_rates[i_active_electrode] > config["params"]["min_hz"]))
                push!(removal_indices, i_active_electrode)
            end
        end
    elseif (dset_id == 3)
        for (i_active_electrode, electrode_name) in enumerate(electrode_names)
            if !((electrode_name[6] == 'a' || electrode_name[6] == 'A') &&
                 (firing_rates[i_active_electrode] > config["params"]["min_hz"]))
                push!(removal_indices, i_active_electrode)
            end
        end
    elseif (dset_id == 4)
        for (i_active_electrode, electrode_name) in enumerate(electrode_names)
            # TODO: how are doubles indicated here?
            if !((firing_rates[i_active_electrode] > config["params"]["min_hz"]))
                push!(removal_indices, i_active_electrode)
            end
        end
    else
        error("Data set type not implemented.")
    end

    # Remove the invalid electrodes from the firing rates, positions etc
    splice!(electrode_names, removal_indices)
    splice!(firing_rates, removal_indices)
    electrode_positions = electrode_positions[
        setdiff(1:size(electrode_positions, 1), removal_indices), :]

    # remove the invalid electrodes from spikes and spike spike_counts
    for (i_removal, removal_idx) in enumerate(removal_indices)
        removal_idx = removal_idx - count(removal_indices[1:i_removal] .< removal_idx)
        st_cumsum = [0; cumsum(spike_counts)]
        st_range = (st_cumsum[removal_idx]+1):(st_cumsum[removal_idx+1])
        splice!(spikes, st_range)
        splice!(spike_counts, removal_idx)
    end

    return electrode_names, electrode_positions, spikes, spike_counts
end

