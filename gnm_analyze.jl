using HDF5
using DataFrames
using Statistics
using Plots

function plot_landscape(df)
    plots = []

    for model in unique(df.model)
        data = filter(row -> row.model == model, df)[:, ["sample", "eta", "gamma", "KS_MAX"]]
        grouped_data = combine(groupby(data, [:eta, :gamma]), :KS_MAX => mean)
        landscape = reshape(grouped_data.KS_MAX_mean,
            (length(unique(data.eta)), length(unique(data.gamma))))
        p = heatmap(landscape,
            clim=(0, 1),
            c=:viridis,
            legend=:none,
            title="Model " * string(model),
            xticks=:none,
            yticks=:none
        )
        yflip!(true)
        push!(plots, p)
    end

    l = @layout[[° ° °; ° ° °; ° ° °; ° ° °; ° _ _] a{0.05w}]
    bar = heatmap((0:0.01:1) .* ones(101, 1), legend=:none, xticks=:none,
        yticks=(1:10:101, string.(0:0.1:1)), c=:viridis)

    plot(plots..., bar, layout=l, size=(1600, 2000))
    savefig("/Users/maxwuerfek/code/diss/jl/test/landscape.png")
end

function analyze(path)
    K, K_max, params = nothing, nothing, nothing
    h5open(path, "r") do file
        K = read(file, "K")
        params = read(file, "params")
        K_max = maximum(K, dims=ndims(K))
        K_max = dropdims(K_max, dims=ndims(K_max))
    end

    df = DataFrame()
    df.sample = repeat(1:size(K, 1), inner=(size(K, 2) * size(K, 3),))
    df.model = repeat(repeat(1:size(K, 2), inner=(size(K, 3),)), size(K, 1))
    df.eta = repeat(params[:, 1], size(K, 1) * size(K, 2))
    df.gamma = repeat(params[:, 2], size(K, 1) * size(K, 2))
    df.KS_K = vec(permutedims(K[:, :, :, 1], [3, 2, 1]))
    df.KS_C = vec(permutedims(K[:, :, :, 2], [3, 2, 1]))
    df.KS_B = vec(permutedims(K[:, :, :, 3], [3, 2, 1]))
    df.KS_E = vec(permutedims(K[:, :, :, 4], [3, 2, 1]))
    df.KS_MAX = vec(permutedims(K_max, [3, 2, 1]))

    plot_landscape(df)
end


function main()
    analyze("/Users/maxwuerfek/code/diss/jl/test/results.h5")
end

main()

