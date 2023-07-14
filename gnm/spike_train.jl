using HDF5
include("sttc.jl")

struct Spike_Train
    spikes::Vector{Float64}
    spike_counts::Vector{Float64}
    num_active_electrodes::Int
    active_electrode_names::Vector{String}
    group_id::String
    recording_time::Vector{Float64}

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
        dt = 0.05
        sttc = sttc_tiling(dt, recording_time, spikes, spike_counts)

        active_electrode_names = read(file, "names")
        active_electrode_names = get_active_electrodes(
            mea_type,
            active_electrode_names,
            all_electrodes
        )


        # read the group id
        if (dset_type == 1)
            age = read(file, "meta/age")[1]
            region = read(file, "meta/region")[1]
            group_id = region * string(age)
        elseif (dset_type == 2)
            # TODO
        end

        close(file)
        new(
            spikes,
            spike_counts,
            length(spike_counts),
            active_electrode_names,
            group_id,
            recording_time
        )
    end
end

function get_active_electrodes(
    mea_type::Int,
    active_electrode_names::Vector{String},
    all_electrodes::Vector{Tuple{Int,Int}})

    removal_indices = []
    active_electrodes = Vector{Tuple{Int,Int}}()
    if (mea_type == 1)
        for (i_active_electrode, active_electrode_name) in enumerate(active_electrode_names)
            if (active_electrode_name[6] == 'a' || active_electrode_name[6] == 'A')
                x = Int(active_electrode_name[4]) - Int('0')
                y = Int(active_electrode_name[5]) - Int('0')
                push!(active_electrodes, (x, y))
            else
                push!(removal_indices, i_active_electrode)
            end
        end

    elseif (mea_type == 2) # TODO
    end

    # remove the invalid electrodes
    splice!(active_electrode_names, removal_indices)

    # find actice indices
    active_idx = findall(in(active_electrodes), all_electrodes)


    return active_electrode_names
end



# p = "/Users/maxwuerfek/code/diss/data/Charlesworth2015/TC190-DIV26_A.h5"
# x = Spike_Train(p, 1)
