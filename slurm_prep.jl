using HDF5
using Printf


function generate_param_space(
    n_runs::Int=100,
    limits_one::Array{Float64}=[-7.0, 7.0],
    limits_two::Array{Float64}=[-7.0, 7.0])
    """
    Creates a linear parameter space defined by the eta and gamma bounds
    for the desired number of runs
    """
    p = range(limits_one[1], stop=limits_one[2], length=max(floor(Int64, sqrt(n_runs)), 2))
    q = range(limits_two[1], stop=limits_two[2], length=max(floor(Int64, sqrt(n_runs)), 2))

    params_tups = unique(collect(Iterators.product(p, q)), dims=1)
    params_tups = vec(permutedims(params_tups, [2, 1]))

    params = zeros(length(params_tups), 2)
    for iParams in 1:length(params_tups)
        params[iParams, 1] = params_tups[iParams][1]
        params[iParams, 2] = params_tups[iParams][2]
    end

    return params
end

function main()
    # params
    n_runs = 4000
    eta_range = [0.0001, 0.3] # [0.8, 1.2] omeg
    gamma_range = [0.5, 1.5]
    params = generate_param_space(n_runs, eta_range, gamma_range)

    println("Params: ", size(params))

    # write files
    sample_i = 1
    for i_param_comb in 1:size(params, 1)
        params_file = params[i_param_comb, 1]
        out_file = "/store/DAMTPEGLEN/mw894/data/weighted/" * "sample_" * @sprintf("%05d", sample_i) * ".h5"
        file = h5open(out_file, "w")

        # write data
        meta_group = create_group(file, "meta")
        attributes(meta_group)["eta"] = params[i_param_comb, 1]
        attributes(meta_group)["gamma"] = params[i_param_comb, 2]
        close(file)

        sample_i += 1
    end

    println("Files: ", sample_i)
end

main()