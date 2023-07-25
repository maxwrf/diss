include("spike_sample.jl")

struct Spike_Dset
    h5_files::Vector{String}
    spike_trains::Vector{Spike_Train}

    function Spike_Dset(
        dir_path::String,
        mea_type::Int,
        dset_type::Int,
        n_samples::Int,
        dt::Float64,
        corr_cutoff::Float64
    )
        sample_dirs = filter(item -> isdir(joinpath(dir_path, item)), readdir(dir_path))

        # construct the spike trains
        if n_samples != -1
            h5_files = h5_files[1:n_samples]
        end

        spike_trains = Vector{Spike_Train}()
        for h5_file in h5_files
            st = Spike_Train(
                joinpath(dir_path, h5_file),
                dset_type,
                mea_type,
                dt,
                corr_cutoff
            )
            push!(spike_trains, st)
        end

        new(h5_files, spike_trains)
    end
end


