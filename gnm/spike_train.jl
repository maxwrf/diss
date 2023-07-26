using HDF5
using LinearAlgebra: triu, Symmetric, diagind
using StatsBase: sample
using BenchmarkTools: @time

include("sttc.jl")
include("jitter.jl")

mutable struct Spike_Train
    file_path::String
    group_id::String
    electrode_names::Vector{String}
    electrode_positions::Matrix{Float64}
    functional_connects::Matrix{Float64}

    # to be filled based on information from all points in time
    A_Y::Union{Matrix{Float64},Nothing}
    A_init::Union{Matrix{Float64},Nothing}
    D::Union{Matrix{Float64},Nothing}

    function Spike_Train(
        file_path::String,
        dset_type::Int,
        mea_type::Int,
        dt::Float64
    )

        # read the h5 data
        file = h5open(file_path, "r")
        spikes = read(file, "spikes")
        spike_counts = read(file, "sCount")
        recording_time = [minimum(spikes), maximum(spikes)]
        electrode_names = read(file, "names")
        electrode_positions = Matrix(read(file, "epos"))
        firing_rates = read(file, "/summary/frate")

        if (dset_type == 1)
            age = read(file, "meta/age")[1]
            region = read(file, "meta/region")[1]
            group_id = region * string(age)
        elseif (dset_type == 3)
            age = read(file, "meta/age")[1]
            group_id = string(age)
        end
        close(file)

        # filter the electrodes
        electrode_names, electrode_positions, spikes, spike_counts = filter_electrodes(
            mea_type,
            electrode_names,
            electrode_positions,
            firing_rates,
            spikes,
            spike_counts
        )

        # functional connectivity inference
        functional_connects = functional_connectivity_inference(
            spikes,
            spike_counts,
            dt,
            recording_time
        )

        # construct the sample
        new(
            file_path,
            group_id,
            electrode_names,
            electrode_positions,
            functional_connects,
            nothing,
            nothing,
            nothing
        )
    end
end

function functional_connectivity_inference(
    spikes::Vector{Float64},
    spike_counts::Vector{Int32},
    dt::Float64,
    recording_time::Vector{Float64}
)::Matrix{Float64}
    # params
    num_permutations = 1000
    p_value = 0.1
    dt_jitter = 0.01

    # compute experimental sttc
    sttc = sttc_tiling(dt, recording_time, spikes, spike_counts)

    # prepare permutations
    jittered_sttc = zeros(num_permutations, size(sttc)...)
    for i in 1:num_permutations
        jittered_spikes = jitter_spikes_fast(spikes, spike_counts, dt_jitter)
        jittered_sttc[i, :, :] = sttc_tiling(dt, recording_time, jittered_spikes, spike_counts)
    end

    # compute p_value
    functional_connects = zeros(size(sttc))
    for i in 1:length(spike_counts)
        for j in 1:length(spike_counts)
            p_val = count(jittered_sttc[:, i, j] .>= sttc[i, j]) / num_permutations
            functional_connects[i, j] = Int(p_val <= p_value)
        end
    end

    return functional_connects
end


function filter_electrodes(
    mea_type::Int,
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
    if (mea_type == 1 || mea_type == 2)
        for (i_active_electrode, electrode_name) in enumerate(electrode_names)
            if !((electrode_name[6] == 'a' || electrode_name[6] == 'A') &&
                 (firing_rates[i_active_electrode] > 0.01))
                push!(removal_indices, i_active_electrode)
            end
        end
    elseif (mea_type == 3)
        # TODO
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

