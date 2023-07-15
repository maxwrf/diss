using HDF5

const MODELS = Dict(
    1 => "spatial",
    2 => "neighbors",
    3 => "matching",
    4 => "clu-avg",
    5 => "clu-min",
    6 => "clu-max",
    7 => "clu-dist",
    8 => "clu-prod",
    9 => "deg-avg",
    10 => "deg-min",
    11 => "deg-max",
    12 => "deg-dist",
    13 => "deg-prod"
)

function ks_test(x, y)
    """
    Computes the Kolmogorov-Smirnov (KS) between two arrays.
    Is a non-parametric test of equality of continous 1D probability 
    distributions from two samples
    A = adjacency matrix
    n = number of nodes
    """
    x_sorted = sort(x)
    y_sorted = sort(y)
    xy_sorted = sort(vcat(x_sorted, y_sorted))

    # Calculate the cumulative distribution functions (CDFs)
    cdf_x = [searchsortedlast(x_sorted, x) for x in xy_sorted] ./ length(x)
    cdf_y = [searchsortedlast(y_sorted, y) for y in xy_sorted] ./ length(y)

    # ks stastistic is the maximum difference
    diff_cdf = abs.(cdf_x - cdf_y)
    ks_statistic = maximum(diff_cdf)

    return ks_statistic
end

function generate_param_space(n_runs::Int=100,
    eta_limits::Array{Float64}=[-7.0, 7.0],
    gamma_limits::Array{Float64}=[-7.0, 7.0])
    """
    Creates a linear parameter space defined by the eta and gamma bounds
    for the desired number of runs
    """
    p = range(eta_limits[1], stop=eta_limits[2], length=max(floor(Int64, sqrt(n_runs)), 2))
    q = range(gamma_limits[1], stop=gamma_limits[2], length=max(floor(Int64, sqrt(n_runs)), 2))

    params_tups = unique(collect(Iterators.product(p, q)), dims=1)
    params_tups = vec(permutedims(params_tups, [2, 1]))

    params = zeros(length(params_tups), 2)
    for iParams in 1:length(params_tups)
        params[iParams, 1] = params_tups[iParams][1]
        params[iParams, 2] = params_tups[iParams][2]
    end

    return params
end

function save_K(dir, K, params)
    h5open(dir * "results.h5", "w") do file
        write(file, "K", K)
        write(file, "params", params)
    end
end
