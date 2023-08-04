using HDF5
using Printf

include("gnm_utils.jl")
include("test_data.jl")

function main()
    # params
    n_runs = 6000
    limits_one = [-7.0, 7.0] #  eta
    limits_two = [-7.0, 7.0] # gamma [7,7]

    limits_three = [0.02, 0.4] # [0.02, 0.1] [0.02, 0.2] alpha
    limits_four = [0.8, 2.0] # [0.8, 1.2] omega

    # load
    W_Y, D, A_init, coords = load_weight_test_data()
    A_Y = Float64.(W_Y .> 0)

    # generate param space
    η = range(limits_one[1], stop=limits_one[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    γ = range(limits_two[1], stop=limits_two[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    α = range(limits_three[1], stop=limits_three[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    ω = range(limits_four[1], stop=limits_four[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))

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
        write(file, "A_Y", A_Y)
        write(file, "D", D)
        write(file, "A_init", A_init)
        write(file, "W_Y", W_Y)
        write(file, "coords", coords)
        write(file, "params", params_file)

        close(file)
        sample_i += 1
    end

    println("Files: ", sample_i)
end

main()