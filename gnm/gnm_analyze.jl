using HDF5
using DataFrames
using Statistics
using Plots

include("gnm_utils.jl")

function plot_landscape(df, group_res_p::String)

    plots = []

    for (model_id, model_name) in MODELS
        data = filter(row -> row.model_id == model_id, df)[:, ["sample", "eta", "gamma", "KS_MAX"]]
        grouped_data = combine(groupby(data, [:eta, :gamma]), :KS_MAX => mean)
        landscape = reshape(grouped_data.KS_MAX_mean,
            (length(unique(data.eta)), length(unique(data.gamma))))
        p = heatmap(landscape,
            clim=(0, 1),
            c=:viridis,
            legend=:none,
            title="Model " * string(model_name),
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

    #save
    out_name = replace(group_res_p, r"\.h5$" => ".png")
    savefig(out_name)
end

function analyze(group_res_p::String)
    """
    Warning: It is important to note that the loaded arrays for each model can 
    have a different numeber of sampels as each sample model combi is run as a 
    single job on the HPC - and can time out.
    """
    file = h5open(group_res_p, "r")

    # read meta data for the group
    meta_group = file["meta"]
    d_set_id = read_attribute(meta_group, "data_set_id")
    data_set_name = read_attribute(meta_group, "data_set_name")
    group_id = read_attribute(meta_group, "group_id")
    param_space = read(file, "param_space")

    # read the data
    results_group = file["results"]
    K_all = []
    for (model_id, _) in MODELS
        # 3D Array: n_sampels x params x 4
        try
            push!(K_all, read(results_group, string(model_id)))
        catch KeyError
            println("Model $model_id not found in $group_res_p.")
        end
    end
    close(file)


    # prepare df of all results
    df_all = []
    for (i_model, K_model) in enumerate(K_all)
        # compute K max
        K_model_max = maximum(K_model, dims=ndims(K_model))
        K_model_max = dropdims(K_model_max, dims=ndims(K_model))

        # prepare df for that model
        df = DataFrame()
        df.sample = repeat(collect(1:size(K_model, 1)), size(K_model, 2))
        df.model_id = repeat([i_model], size(K_model, 1) * size(K_model, 2))
        df.eta = repeat(param_space[:, 1], size(K_model, 1))
        df.gamma = repeat(param_space[:, 2], size(K_model, 1))
        df.KS_K = vec(permutedims(K_model[:, :, 1], [2, 1]))
        df.KS_C = vec(permutedims(K_model[:, :, 2], [2, 1]))
        df.KS_B = vec(permutedims(K_model[:, :, 3], [2, 1]))
        df.KS_E = vec(permutedims(K_model[:, :, 4], [2, 1]))
        df.KS_MAX = vec(permutedims(K_model_max, [2, 1]))

        push!(df_all, df)
    end
    df_all = vcat(df_all...)

    plot_landscape(df_all, group_res_p)
end


