using Distances
include("spike_train.jl")

struct Spike_Set
    h5_files::Vector{String}
    spike_trains::Vector{Spike_Train}
    D::Matrix{Float64}

    function Spike_Set(
        dir_path::String,
        mea_type::Int,
        dset_type::Int,
        n_samples::Int,
        dt::Float64,
        corr_cutoff::Float64
    )
        h5_files = filter(name -> endswith(name, ".h5"), readdir(dir_path))

        # construct all info that is consistenrt across all spike trains
        all_electrodes, all_electrode_pos = get_electode_positions(mea_type)
        D = pairwise(Euclidean(), all_electrode_pos, all_electrode_pos)

        # construct the spike trains
        if n_samples != -1
            h5_files = h5_files[1:n_samples]
        end

        spike_trains = Vector{Spike_Train}()
        for h5_file in h5_files
            push!(spike_trains,
                Spike_Train(
                    joinpath(dir_path, h5_file),
                    mea_type,
                    dset_type,
                    all_electrodes,
                    dt,
                    corr_cutoff
                )
            )
        end

        new(h5_files, spike_trains, D)
    end
end

function get_electode_positions(mea_type::Int)
    # load all data specific to the mea
    if mea_type == 1 # MCS_8x8_200um
        row_num_electrodes = 8
        electrode_dist = 200
        from_top_right = true
        exclude_electrodes = [(1, 1), (1, 8), (8, 1), (8, 8)]
        start_dist_multiplier = 1
    elseif mea_type == 2 # MCS_8x8_100um
        row_num_electrodes = 8
        electrode_dist = 200
        from_top_right = true
        exclude_electrodes = [(1, 1), (1, 8), (8, 1), (8, 8)]
        start_dist_multiplier = 1
    elseif mea_type == 3 # APS_64x64_42um
        row_num_electrodes = 64
        electrode_dist = 42
    else
        error("mea_type not supported")
    end

    # construct the electrode positions
    electrodes = Vector{Tuple{Int,Int}}()
    electrode_pos = Vector{Tuple{Int,Int}}()

    for i in 1:row_num_electrodes
        for j in 1:row_num_electrodes
            if (i, j) ∉ exclude_electrodes
                push!(electrodes, (i, j))
                x = i * electrode_dist
                y = from_top_right ? ((row_num_electrodes + 1) - j) * electrode_dist : j * electrode_dist
                push!(electrode_pos, (x, y))
            end
        end
    end

    return electrodes, electrode_pos
end