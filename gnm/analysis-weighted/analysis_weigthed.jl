using DataFrames
using HDF5
using Plots
using Measures
using StatsBase

include("../test_data.jl")
include("../graph_utils.jl")

function load_files()
    # load generate
    res_files = filter(file -> endswith(file, ".res"), readdir("/store/DAMTPEGLEN/mw894/data/weighted"))
    df_all = DataFrame[]

    for (i_res_files, res_file) in enumerate(res_files)

        file = h5open(joinpath("/store/DAMTPEGLEN/mw894/data/weighted", res_file), "r")
        df_file = DataFrame()

        # read meta data
        meta_group = file["meta"]
        params = read_attribute(meta_group, "params")
        file_path = read_attribute(meta_group, "file_name")

        # read data
        K_pcomb = read(file, "K_pcomb")         # 4
        K_W_pcomb = read(file, "K_W_pcomb")     # 3

        println("File: ", res_file, "params: ", params)


        # make df
        df_file.file_name = [file_path]

        df_file.KS_K = [K_pcomb[1]]
        df_file.KS_C = [K_pcomb[2]]
        df_file.KS_B = [K_pcomb[3]]
        df_file.KS_E = [K_pcomb[4]]

        df_file.KS_S = [K_W_pcomb[1]]
        df_file.KS_WC = [K_W_pcomb[2]]
        df_file.KS_WB = [K_W_pcomb[3]]

        df_file.eta = [params[1]]
        df_file.gamma = [params[2]]
        df_file.alpha = [params[3]]
        df_file.omega = [params[4]]

        push!(df_all, df_file)
    end
    df_all = vcat(df_all...)

    df_all.KS_MAX = map(maximum, eachrow(df_all[!, ["KS_B", "KS_C", "KS_E", "KS_K"]]))
    df_all.KS_W_MAX = map(maximum, eachrow(df_all[!, ["KS_S", "KS_WC", "KS_WB"]]))

    return df_all
end

function landscape_plot(df)
    # best KS for eta gamma
    df_eta_gamma = combine(groupby(df, [:eta, :gamma]), :KS_MAX => minimum => :KS_MAX, :KS_W_MAX => minimum => :KS_W_MAX)
    sort!(df_eta_gamma, [:eta, :gamma])

    landscape = reshape(df_eta_gamma.KS_MAX, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p1 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="KS max (best eta gamma)",
        clim=(0, 1), c=:viridis)

    # best KS W for eta gamma
    landscape = reshape(df_eta_gamma.KS_W_MAX, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p2 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="KS W max (best eta gamma)",
        clim=(0, 1), c=:viridis)

    # best KS W for eta gamma
    df_alpha_omega = combine(groupby(df, [:alpha, :omega]), :KS_MAX => minimum => :KS_MAX, :KS_W_MAX => minimum => :KS_W_MAX)
    sort!(df_alpha_omega, [:alpha, :omega])

    landscape = reshape(df_alpha_omega.KS_W_MAX, (length(unique(df_alpha_omega.alpha)), length(unique(df_alpha_omega.omega))))
    p3 = heatmap(unique(df_alpha_omega.alpha), unique(df_alpha_omega.omega), landscape, xlabel="alpha", ylabel="omega", title="KS W max (best alpha omega)",
        clim=(0, 1), c=:viridis)

    p = plot(p1, p2, p3;
        layout=grid(2, 2),
        fmt=:pdf,
        size=(1500, 1400),
        margin=8mm)

    savefig(p, "/home/mw894/diss/gnm/analysis-weighted/w_landscapes.pdf")
end

function dist_plot_W(df)
    # load y
    W_Y, D, A_init, coord = load_weight_test_data()
    A_Y = Float64.(W_Y .> 0)
    S_Y = dropdims(sum(weight_conversion(W_Y), dims=1), dims=1)
    WC_Y = dropdims(clustering_coef_wu(weight_conversion(W_Y)), dims=2)
    WB_Y = betweenness_wei(weight_conversion(W_Y))
    dists_Y = [S_Y, WC_Y, WB_Y]

    # pred
    best_energy = df[argmin(df.KS_W_MAX), :]
    file = h5open(best_energy.file_name, "r")
    W_pred = read(file, "W_final_pcomb")
    close(file)
    S_Y = dropdims(sum(weight_conversion(W_pred), dims=1), dims=1)
    WC_Y = dropdims(clustering_coef_wu(weight_conversion(W_pred)), dims=2)
    WB_Y = betweenness_wei(weight_conversion(W_pred))
    dists_Y_pred = [S_Y, WC_Y, WB_Y]

    plots = []
    for (e_Y, e_Y_pred) in zip(dists_Y, dists_Y_pred)
        cdf_y = ecdf(e_Y)
        cdf_y_pred = ecdf(e_Y_pred)

        # y
        x = sort(unique(e_Y))
        p = plot(x, cdf_y(x),
            linewidth=3,
            label="CDF",
            xlabel="Values",
            ylabel="Cumulative Probability",
            title="Cumulative Distribution Function")

        #pred
        x = sort(unique(e_Y_pred))
        p = plot!(x, cdf_y_pred(x), linewidth=3, label="CDF pred")

        push!(plots, p)
    end

    p = plot(
        plots...,
        layout=(2, 2),
        size=(1500, 1500),
        fmt=:pdf,
        title=reshape(["Strength", "Weighted clustering", "Weighted betweness"], (1, 3)),
        margin=5mm
    )
    savefig(p, "/home/mw894/diss/gnm/analysis-weighted/cum_dist_W.pdf")
end

function dist_plot_A(df)
    # load y
    W_Y, D, A_init, coord = load_weight_test_data()
    A_Y = Float64.(W_Y .> 0)

    energy_Y = zeros(4, size(A_Y, 1))

    energy_Y[1, :] = sum(A_Y, dims=1)
    energy_Y[2, :] = get_clustering_coeff(A_Y, size(A_Y, 1))
    energy_Y[3, :] = betweenness_centrality(A_Y, size(A_Y, 1))
    energy_Y[4, :] = sum((D .* A_Y), dims=1)

    # pred
    best_energy = df[argmin(df.KS_MAX), :]
    file = h5open(best_energy.file_name, "r")
    A_pred = read(file, "A_final_pcomb")
    close(file)

    energy_Y_head = zeros(4, size(A_pred, 1))
    energy_Y_head[1, :] = sum(A_pred, dims=1)
    energy_Y_head[2, :] = get_clustering_coeff(A_pred, size(A_pred, 1))
    energy_Y_head[3, :] = betweenness_centrality(A_pred, size(A_pred, 1))
    energy_Y_head[4, :] = sum((D .* A_pred), dims=1)

    plots = []
    for i in 1:4
        cdf_y = ecdf(energy_Y[i, :])
        cdf_y_pred = ecdf(energy_Y_head[i, :])

        # y
        x = sort(unique(energy_Y[i, :]))
        p = plot(x, cdf_y(x),
            linewidth=3,
            label="CDF",
            xlabel="Values",
            ylabel="Cumulative Probability",
            title="Cumulative Distribution Function")

        #pred
        x = sort(unique(energy_Y_head[i, :]))
        p = plot!(x, cdf_y_pred(x), linewidth=3, label="Pred CDF")

        push!(plots, p)
    end

    p = plot(
        plots...,
        layout=(2, 2),
        size=(1500, 1500),
        fmt=:pdf,
        title=reshape(["Degree", "Clustering", "Betwennes", "Edege length"], (1, 4)),
        margin=5mm
    )
    savefig(p, "/home/mw894/diss/gnm/analysis-weighted/cum_dist_A.pdf")
end

df_all = load_files()
landscape_plot(df_all)
dist_plot_A(df_all)
dist_plot_W(df_all)

x = df_all[argmin(df_all.KS_MAX), :]
y = df_all[argmin(df_all.KS_W_MAX), :]


# W_Y, D, A_init, coord = load_weight_test_data()
# A_Y = Float64.(W_Y .> 0)
# S_Y = dropdims(sum(weight_conversion(W_Y), dims=1), dims=1)
# WC_Y = dropdims(clustering_coef_wu(weight_conversion(W_Y)), dims=2)
# WB_Y = betweenness_wei(weight_conversion(W_Y))