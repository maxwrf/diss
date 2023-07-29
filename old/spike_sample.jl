using Distances: Euclidean, pairwise

include("spike_train.jl")

struct Spike_Sample
    h5_files::Vector{String}
    spike_trains::Vector{Spike_Train}

    function Spike_Sample(
        dir_path::String,
        mea_type::Int,
        dt::Float64
    )
        h5_files = filter(name -> endswith(name, ".h5"), readdir(dir_path))

        spike_trains = Vector{Spike_Train}()
        for h5_file in h5_files
            st = Spike_Train(
                joinpath(dir_path, h5_file),
                mea_type,
                dt
            )
            push!(spike_trains, st)
        end

        prepare_sample(spike_trains)

        new(
            h5_files,
            spike_trains
        )
    end
end

function prepare_sample(spike_trains::Vector{Spike_Train})
    # concatenate the electrode names and positions
    sample_electrode_names = vcat([st.electrode_names for st in spike_trains]...)
    sample_electrode_positions = vcat([st.electrode_positions for st in spike_trains]...)

    # get the unique sample electrode names and positions
    uniq_electrode_idx = unique(i -> sample_electrode_names[i], 1:length(sample_electrode_names))
    sample_electrode_names = sample_electrode_names[uniq_electrode_idx]
    sample_electrode_positions = sample_electrode_positions[uniq_electrode_idx, :]

    # compute the distance matrix
    D = pairwise(Euclidean(), sample_electrode_positions, dims=1)

    for st in spike_trains
        # find the electrodes of that st in the sample
        electrode_idx = findall(e_name -> e_name in sample_electrode_names, st.electrode_names)
        electrode_idx = [CartesianIndex(i, j) for i in electrode_idx, j in electrode_idx]

        # initalize the adjacency matrices
        A_Y = zeros(size(D))
        A_init = zeros(size(D))

        # paste the functional elements into the A 
        A_Y[electrode_idx] = st.functional_connects

        # compute the adjacency matrices
        A_Y[diagind(A_Y)] .= 0
        m = sum(A_Y) / 2

        # prepare the initalization matrix
        edges = findall(==(1), triu(A_Y, 1))
        init_edges = sample(edges, Int(round(m * 0.2)); replace=false)
        A_init[init_edges] .= 1
        A_init = Symmetric(A_init, :U)

        # set on the spike train
        st.A_Y = A_Y
        st.A_init = A_init
        st.D = D
    end
end