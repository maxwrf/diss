using HDF5
using DataFrames
using Statistics
using Plots
using Measures

Plots.scalefontsizes(1.5)

include("/home/mw894/diss/gnm/gnm_utils.jl")

datasets = [
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx",
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hpc",
    "/store/DAMTPEGLEN/mw894/data/Demas2006"
]

function get_df_all()
    df_all = DataFrame[]
    for in_dir in datasets
        res_files = filter(name -> endswith(name, ".res"), readdir(in_dir))
        res_files = map(name -> joinpath(in_dir, name), res_files)
        res_files

        df_dataset = DataFrame[]

        # Each result stores for one sample and one model
        for (i_res_files, res_file) in enumerate(res_files)
            file = h5open(res_file, "r")
            df_file = DataFrame()

            # read meta data for this sample model combi
            meta_group = file["meta"]
            div = read_attribute(meta_group, "group_id")
            sample_name = read_attribute(meta_group, "org_file_name")
            data_set_name = read_attribute(meta_group, "data_set_name")
            model_id = read_attribute(meta_group, "model_id")
            week = min(4, ceil.(Int, parse(Int, div) / 7))

            # read the data for this sample model combi
            K = read(file, "K")

            # read the paramter space
            param_space = read(file, "param_space")
            n_rows = size(param_space, 1)
            close(file)

            # store metadata for df
            df_file.model_id = repeat([model_id], n_rows)
            df_file.sample_name = repeat([sample_name], n_rows)
            df_file.div = repeat([div], n_rows)
            df_file.week = repeat([week], n_rows)
            df_file.data_set = repeat([data_set_name], n_rows)
            df_file.eta = param_space[:, 1]
            df_file.gamma = param_space[:, 2]
            df_file.KS_K = K[:, 1]
            df_file.KS_C = K[:, 2]
            df_file.KS_B = K[:, 3]
            df_file.KS_E = K[:, 4]

            push!(df_dataset, df_file)
        end

        df_dataset = vcat(df_dataset...)
        df_dataset.KS_MAX = map(maximum, eachrow(df_dataset[!, ["KS_B", "KS_C", "KS_E", "KS_K"]]))

        push!(df_all, df_dataset)
    end

    df_all = vcat(df_all...)

    return df_all
end

function get_overall_heatmap()
    plots = []
    dset_names = []
    for dset in unique(df_all.data_set)
        push!(dset_names, dset)

        # get dataset df
        df_dataset = df_all[df_all.data_set.==dset, :]

        # Get best performing parameter combination 
        top_model_sample_combs = combine(groupby(df_dataset, [:model_id, :sample_name, :week]), :KS_MAX => minimum => :KS_MAX_best)

        # Compute the average across all samples
        avg_top_model_sample_combs = combine(groupby(top_model_sample_combs, [:model_id, :week]), :KS_MAX_best => mean => :KS_MAX_best_mean)

        # data for plot
        p_data = reshape(avg_top_model_sample_combs.KS_MAX_best_mean,
            length(unique(avg_top_model_sample_combs.week)),
            length(unique(avg_top_model_sample_combs.model_id)))'


        p = heatmap(p_data, yticks=(1:13, values(MODELS)), interpolate=false, c=:viridis, xticks=(1:4), fill_z=p_data, fmt=:pdf)

        # annotate
        nrow, ncol = size(p_data)
        fontsize = 11
        ann = [(j, i, text(round(p_data[i, j], digits=3), fontsize, :white, :center, :bold)) for i in 1:nrow for j in 1:ncol]
        annotate!(ann, linecolor=:white)
        xlabel!("DIV in number of weeks")

        push!(plots, p)
    end

    p = plot(plots...; format=grid(4, 4), fmt=:pdf, size=(1500, 1500), margin=5mm, title=reshape(dset_names, (1, 3)))
    savefig(p, "gnm/analysis/top_scores.pdf")
end

function get_freq_heatmap()
    plots = []
    dset_names = []
    for dset in unique(df_all.data_set)
        push!(dset_names, dset)


        # get dataset df
        df_dataset = df_all[df_all.data_set.==dset, :]

        # get top parm_combs
        top_param_combs = combine(groupby(df_dataset, [:model_id, :sample_name, :week]), :KS_MAX => minimum => :KS_MAX_best)

        # For each sample get the best model
        tops = combine(g -> g[argmin(g.KS_MAX_best), :], groupby(top_param_combs, [:sample_name, :week]))

        # For each model count the samples
        tops = combine(groupby(tops, [:model_id, :week]), :KS_MAX_best => length => :nrow)

        # combine because we dropped out
        product = Iterators.product(1:13, 1:4)
        df_plot = DataFrame(product)
        rename!(df_plot, [:model_id, :week])
        df_plot = leftjoin(df_plot, tops, on=[:model_id, :week])
        replace!(df_plot.nrow, missing => 0)
        sort!(df_plot, [:model_id, :week])

        # data for plot
        p_data = reshape(df_plot.nrow,
            length(unique(df_plot.model_id)),
            length(unique(df_plot.week)))

        p_data = p_data ./ sum(p_data, dims=1)


        p = heatmap(p_data, yticks=(1:13, values(MODELS)), interpolate=false, xticks=(1:4), fill_z=p_data, fmt=:pdf)

        # annotate
        nrow, ncol = size(p_data)
        fontsize = 11
        ann = [(j, i, text(round(p_data[i, j], digits=3), fontsize, :white, :center, :bold)) for i in 1:nrow for j in 1:ncol]
        annotate!(ann, linecolor=:white)
        xlabel!("DIV in number of weeks")

        push!(plots, p)
    end
    p = plot(plots...; format=grid(4, 4), fmt=:pdf, size=(1500, 1500), margin=5mm, title=reshape(dset_names, (1, 3)))
    savefig(p, "gnm/analysis/top_freqs.pdf")
end

get_freq_heatmap()

function get_heatmaps()
    heatmaps = Dict()
    for dset in unique(df_all.data_set)
        heatmaps[dset] = Dict()
        for week in unique(df_all.week)
            # prep plots and names
            plots = []

            # Demas does not have week 4
            if (week == 4) && (dset == "Demas2006")
                continue
            end

            for (model_id, model_name) in MODELS
                # get model df
                df_subset = filter(r -> (r.model_id == model_id) && (r.week == week) && (r.data_set == dset), df_all)

                # average across samples
                plot_data = combine(groupby(df_subset, [:eta, :gamma]), :KS_MAX => mean)

                # data for plot
                landscape = reshape(plot_data.KS_MAX_mean, (length(unique(plot_data.eta)), length(unique(plot_data.gamma))))
                p = heatmap(landscape, clim=(0, 1), c=:viridis, legend=:none, title="Model " * string(model_name), xticks=:none, yticks=:none)
                yflip!(true)
                push!(plots, p)
            end

            println("heatmaps", dset, " ", week)

            # combine and add bar
            push!(plots, plot())
            push!(plots, plot())
            push!(plots, plot())
            l = @layout[grid(4, 4) a{0.05w}]
            bar = heatmap((0:0.01:1) .* ones(101, 1), legend=:none, xticks=:none, yticks=(1:10:101, string.(0:0.1:1)), c=:viridis)
            p = plot(plots..., bar, layout=l, size=(2000, 1600))

            heatmaps[dset][week] = plots

            savefig(p, "gnm/analysis/" * replace(dset, "/" => "_") * "_" * string(week) * "_heatmaps.png")
        end
    end
    return heatmaps
end

function get_top_df()
    # groupby datset and week and get the best performing parameter combination
    tops = combine(g -> g[argmin(g.KS_MAX), :], groupby(df_all, [:data_set, :model_id, :sample_name, :week]))

    # Compute the average across all samples
    avg_tops = combine(groupby(tops, [:data_set, :model_id, :week]),
        :KS_MAX => mean => :KS_MAX_best_mean,
        :eta => mean => :eta_mean,
        :gamma => mean => :gamma_mean)

    top_df = combine(g -> g[argmin(g.KS_MAX_best_mean), :], groupby(avg_tops, [:data_set, :week]))

    sort!(top_df, [:data_set, :week])

    return top_df
end


get_top_df()

get_overall_heatmap()

heatmaps = get_heatmaps()

df_all = get_df_all()


