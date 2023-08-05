using HDF5
using Printf

include("gnm_utils.jl")
include("test_data.jl")

function main()
    # params
    n_runs = 4000

    gamma_three = [0.01, 0.2] # [0.02, 0.1] [0.02, 0.2] alpha
    eta_four = [0.5, 1.5] # [0.8, 1.2] omega

    # load

    # generate param space
    η = range(limits_one[1], stop=limits_one[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    γ = range(limits_two[1], stop=limits_two[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))


    params_tups = unique(collect(Iterators.product(η, γ, α, ω)), dims=1)
    params_tups = vec(permutedims(params_tups, [4, 3, 2, 1]))

    params = zeros(length(params_tups), 4)
    for iParams in 1:length(params_tups)
        params[iParams, 1] = params_tups[iParams][1]
        params[iParams, 2] = params_tups[iParams][2]
        params[iParams, 3] = params_tups[iParams][3]
        params[iParams, 4] = params_tups[iParams][4]
    end

    println("Params: ", size(params))

    # write files
    sample_i = 1
    for i_param_comb in 1:size(params, 1)
        params_file = params[i_param_comb, :]
        out_file = "/store/DAMTPEGLEN/mw894/data/weighted/" * "sample_" * @sprintf("%05d", sample_i) * ".h5"
        file = h5open(out_file, "w")

        # write data

        close(file)
        sample_i += 1
    end

    println("Files: ", sample_i)
end

main()