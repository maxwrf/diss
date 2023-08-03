using HDF5
using DataFrames
using Statistics
using Plots
using Measures
using GLM
using LaTeXStrings
using StatsBase

using StatsPlots
using Plots

# Plots.scalefontsizes(1.5)

include("/home/mw894/diss/gnm/gnm_utils.jl")

datasets = [
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx",
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hpc",
    "/store/DAMTPEGLEN/mw894/data/Demas2006"
]

function get_df_all(subsample=false)
    df_all = DataFrame[]
    for in_dir in datasets
        if subsample
            res_files = filter(name -> endswith(name, ".subres"), readdir(in_dir))
        else
            res_files = filter(name -> endswith(name, ".res"), readdir(in_dir))
        end
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

    # remove zeros
    df_all = df_all[df_all.KS_MAX.!=0, :]

    families = Dict(
        1 => "spatial",
        2 => "homophilic",
        3 => "homophilic",
        4 => "cluster",
        5 => "cluster",
        6 => "cluster",
        7 => "cluster",
        8 => "cluster",
        9 => "degree",
        10 => "degree",
        11 => "degree",
        12 => "degree",
        13 => "degree"
    )

    model_names = Dict(
        1 => "spatial",
        2 => "match",
        3 => "neigh",
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

    df_all.families = map(id -> families[id], df_all.model_id)
    df_all.model_names = map(id -> model_names[id], df_all.model_id)

    return df_all
end

function get_overall_heatmap(subsample=false)
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
        sort!(avg_top_model_sample_combs, [:week, :model_id])

        # data for plot
        p_data = reshape(avg_top_model_sample_combs.KS_MAX_best_mean,
            length(unique(avg_top_model_sample_combs.model_id)),
            length(unique(avg_top_model_sample_combs.week)))


        p = heatmap(p_data, yticks=(1:13, values(MODELS)), interpolate=false, c=:viridis, xticks=(1:4), fill_z=p_data, fmt=:pdf, legend=:none)

        # annotate
        nrow, ncol = size(p_data)
        fontsize = 12
        ann = [(j, i, text(round(p_data[i, j], digits=3), fontsize, :white, :center, :bold)) for i in 1:nrow for j in 1:ncol]
        annotate!(ann, linecolor=:white)
        xlabel!("DIV in number of weeks")

        push!(plots, p)
    end

    p = plot(plots...; format=grid(4, 4), fmt=:pdf, size=(1500, 1500), margin=5mm, title=reshape(dset_names, (1, 3)))

    if subsample
        savefig(p, "gnm/analysis/top_scores_sub.pdf")
    else
        savefig(p, "gnm/analysis/top_scores.pdf")
    end

end

function get_freq_heatmap(subsample=false)
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
        if dset == "Demas2006"
            product = Iterators.product(1:13, 1:3)
        else
            product = Iterators.product(1:13, 1:4)
        end
        df_plot = DataFrame(product)
        rename!(df_plot, [:model_id, :week])
        df_plot = leftjoin(df_plot, tops, on=[:model_id, :week])
        replace!(df_plot.nrow, missing => 0)
        sort!(df_plot, [:week, :model_id])

        # data for plot
        p_data = reshape(df_plot.nrow,
            length(unique(df_plot.model_id)),
            length(unique(df_plot.week)))

        p_data = p_data ./ sum(p_data, dims=1)


        p = heatmap(p_data, yticks=(1:13, values(MODELS)), interpolate=false, xticks=(1:4), fill_z=p_data, fmt=:pdf, legend=:none, c=:RdYlBu_3)

        # annotate
        nrow, ncol = size(p_data)
        fontsize = 12
        ann = [(j, i, text(round(p_data[i, j], digits=3), fontsize, :black, :center, :bold)) for i in 1:nrow for j in 1:ncol]
        annotate!(ann, linecolor=:white)
        xlabel!("DIV in number of weeks")

        push!(plots, p)
    end
    p = plot(plots...; format=grid(4, 4), fmt=:pdf, size=(1500, 1500), margin=5mm, title=reshape(dset_names, (1, 3)))

    if subsample
        savefig(p, "gnm/analysis/top_freqs_sub.pdf")
    else
        savefig(p, "gnm/analysis/top_freqs.pdf")
    end
end

function get_heatmaps(subsample=false)
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

            if subsample
                savefig(p, "gnm/analysis/" * replace(dset, "/" => "_") * "_" * string(week) * "_heatmaps_sub.pdf")
            else
                savefig(p, "gnm/analysis/" * replace(dset, "/" => "_") * "_" * string(week) * "_heatmaps.pdf")
            end
        end
    end
    return heatmaps
end

function get_top_freq_df()
    # groupby datset and week and get the best performing parameter combination
    tops = combine(g -> g[argmin(g.KS_MAX), :], groupby(df_all, [:data_set, :sample_name, :week, :model_id]))

    # across models get the best model for each sample
    tops = combine(g -> g[argmin(g.KS_MAX), :], groupby(tops, [:data_set, :sample_name, :week]))

    # For each model count the samples
    tops = combine(groupby(tops, [:data_set, :week, :model_id]), :KS_MAX => length => :ntop, :eta => mean => :mean_eta, :gamma => mean => :mean_gamma)

    # get freqs
    freqs = combine(groupby(tops, [:data_set, :week]), :ntop => sum => :nsamples)

    # get maximum count
    tops = combine(g -> g[argmax(g.ntop), :], groupby(tops, [:data_set, :week]))

    tops = innerjoin(tops, freqs; on=[:data_set, :week])
    tops.fractop = tops.ntop ./ tops.nsamples
    sort!(tops, [:data_set, :week])
    return tops
end

function get_top_df()
    # groupby datset and week and get the best performing parameter combination
    tops = combine(g -> g[argmin(g.KS_MAX), :], groupby(df_all, [:data_set, :model_id, :sample_name, :week]))

    # Compute the average across all samples
    avg_tops = combine(groupby(tops, [:data_set, :model_id, :week]),
        :KS_MAX => mean => :KS_MAX_best_mean,
        :KS_MAX => median => :KS_MAX_best_median,
        :KS_MAX => std => :KS_MAX_best_std,
        :KS_MAX => iqr => :KS_MAX_best_iqr,
        :eta => mean => :eta_mean,
        :gamma => mean => :gamma_mean)

    top_df = combine(g -> g[argmin(g.KS_MAX_best_mean), :], groupby(avg_tops, [:data_set, :week]))
    top_df_median = combine(g -> g[argmin(g.KS_MAX_best_median), :], groupby(avg_tops, [:data_set, :week]))

    #top_df = outerjoin(top_df, top_df2; on=[:data_set, :week], makeunique=true)

    sort!(top_df, [:data_set, :week])
    sort!(top_df_median, [:data_set, :week])


    cols = [:KS_MAX_best_mean, :KS_MAX_best_median, :KS_MAX_best_std, :KS_MAX_best_iqr, :eta_mean, :gamma_mean]

    top_df[!, cols] = round.(top_df[!, cols], digits=3)
    top_df_median[!, cols] = round.(top_df_median[!, cols], digits=3)

    return top_df, top_df_median
end

function top_eta_gamma_combs(df_top, subsample=false)
    plots = []
    titles = []

    i = 0
    i_color = 1
    for row in eachrow(df_top)
        i += 1
        # get the best performing model, per week and dataset
        df_subset = filter(r -> (r.model_id == row.model_id) && (r.week == row.week) && (r.data_set == row.data_set), df_all)

        # for every sample get the best performing parameter combination
        df_subset = combine(g -> g[argmin(g.KS_MAX), :], groupby(df_subset, [:sample_name]))

        # increase color index
        if (i - 1) % 4 == 0
            i_color += 1
        end

        # prep plot
        p = scatter(df_subset.eta, df_subset.gamma, seriestype=:scatter, legend=:none, title=string(row.data_set, " ", row.week),
            aspect_ratio=:equal, color=palette(:default)[i_color], markerstrokecolor=palette(:default)[i_color],
            markersize=10) #limits=(-7.5, 6), xlimits=(-7.5, 2.5),

        # add fitted line
        m = lm(@formula(gamma ~ eta), df_subset)
        Plots.abline!(coef(m)[2], coef(m)[1], color=palette(:default)[i_color])

        annotate!(-5.5, 0.1, "R2: " * string(round(r2(m), digits=2)), 12)


        if (i - 1) % 4 == 0
            ylabel!(row.data_set * "\ngamma")
        end

        if i > 8
            xlabel!("eta")
        end

        push!(plots, p)
        push!(titles, "DIV W" * string(row.week) * "\n Model: " * MODELS[row.model_id])
    end

    p = plot(plots...; format=grid(3, 4), fmt=:pdf, size=(1500, 1500), margin=7mm, title=reshape(titles, (1, 11)))

    if subsample
        savefig(p, "gnm/analysis/eta_gamma_stab_sub.pdf")
    else
        savefig(p, "gnm/analysis/eta_gamma_stab.pdf")
    end
end

function top_landscapes(df_top, subsample=false)
    plots = []
    titles = []

    i = 0
    i_color = 1
    for row in eachrow(df_top)
        i += 1
        # get the best performing model, per week and dataset
        df_subset = filter(r -> (r.model_id == row.model_id) && (r.week == row.week) && (r.data_set == row.data_set), df_all)

        # average across samples
        plot_data = combine(groupby(df_subset, [:eta, :gamma]), :KS_MAX => median => :KS_MAX_mean)

        # increase color index
        if (i - 1) % 4 == 0
            i_color += 1
        end

        # data for plot
        landscape = reshape(plot_data.KS_MAX_mean, (length(unique(plot_data.eta)), length(unique(plot_data.gamma))))
        p = heatmap(landscape, clim=(0, 1), c=:viridis, legend=:none, xticks=:none, yticks=:none, aspect_ratio=:equal, showaxis=false)
        yflip!(true)

        if (i - 1) % 4 == 0
            ylabel!(row.data_set * "\ngamma (-7.5, +7.5)")
        end

        if i > 8
            xlabel!("eta (-7.5, +7.5)")
        end

        push!(plots, p)
        push!(titles, "DIV W" * string(row.week) * "\n Model: " * MODELS[row.model_id])
    end

    p = plot(plots...; format=grid(3, 4), fmt=:pdf, size=(1600, 1300), margin=9.5mm, title=reshape(titles, (1, 11)))

    if subsample
        savefig(p, "gnm/analysis/top_landscapes_sub.pdf")
    else
        savefig(p, "gnm/analysis/top_landscapes.pdf")
    end
end

function top_energy_factor(df_top, subsample=false)
    plots = []
    titles = []

    i = 0
    i_color = 1
    for row in eachrow(df_top)
        i += 1
        # get the best performing model, per week and dataset
        df_subset = filter(r -> (r.model_id == row.model_id) && (r.week == row.week) && (r.data_set == row.data_set), df_all)

        # for every sample get the best performing parameter combination
        df_subset = combine(g -> g[argmin(g.KS_MAX), :], groupby(df_subset, [:sample_name]))

        # increase color index
        if (i - 1) % 4 == 0
            i_color += 1
        end

        # compute the realtive frequencies
        top_energies = reduce(append!, map(r -> findall([r.KS_K, r.KS_C, r.KS_B, r.KS_E] .== r.KS_MAX), eachrow(df_subset)))
        top_energies_counts = countmap(top_energies)
        top_energies_relative_freqs = Dict(key => count / length(top_energies) for (key, count) in top_energies_counts)

        # prep plot
        x = keys(top_energies_relative_freqs)
        y = values(top_energies_relative_freqs)
        order = sortperm(collect(x))
        y = collect(y)[order]
        x = collect(x)[order]

        p = bar(x, y, l=0.5, xticks=(1:4, ["KS K", "KS C", "KS B", "KS E"]), legend=false, ylim=(0, 0.5),
            c=palette(:default)[i_color], title=string(row.data_set, " ", row.week))

        if (i - 1) % 4 == 0
            ylabel!(row.data_set * "\n% maximum energy (KS max)")
        end

        if i > 8
            xlabel!("Energy factor")
        end

        push!(plots, p)
        push!(titles, "DIV W" * string(row.week) * "\n Model: " * MODELS[row.model_id])
    end

    p = plot(plots...; format=grid(3, 4), fmt=:pdf, size=(1500, 1500), margin=7mm, title=reshape(titles, (1, 11)))

    if subsample
        savefig(p, "gnm/analysis/energy_factors_sub.pdf")
    else
        savefig(p, "gnm/analysis/energy_factors.pdf")
    end
end

function get_overall_boxplots(subsample=false)
    plots = [] # 11 plots
    titles = []
    for dset in unique(df_all.data_set)
        for week in unique(df_all.week)
            # Demas does not have week 4
            if (week == 4) && (dset == "Demas2006")
                continue
            end

            # get subset
            df_subset = filter(r -> (r.week == week) && (r.data_set == dset), df_all)

            # Get best performing parameter combination 
            df_subset_top = combine(groupby(df_subset, [:model_names, :sample_name, :week]), :KS_MAX => minimum => :KS_MAX_best, :families)


            p = @df df_subset_top boxplot(:model_names, group=:families, :KS_MAX_best, xrotation=45, fillalpha=0.75, linewidth=2, legend=false, outliers=false)
            push!(plots, p)
            push!(titles, dset * "_w " * string(week))
        end
    end

    p = plot(plots...; layout=grid(3, 4), size=(1500, 1500), fmt=:pdf, margin=5mm, title=reshape(titles, (1, 11)))

    if subsample
        savefig(p, "gnm/analysis/boxplots_sub.pdf")
    else
        savefig(p, "gnm/analysis/boxplots.pdf")
    end
end

subsample = true
# get dfs
df_all = get_df_all(subsample)
df_top, df_top_median = get_top_df()
df_top_freq = get_top_freq_df()

# overall performance
get_overall_boxplots(subsample)
get_overall_heatmap(subsample)

# eta gamma sensitivity
top_eta_gamma_combs(df_top_median, subsample)

Plots.scalefontsizes(1.25)
top_landscapes(df_top_median)

# Sensitivtiy to samples
get_freq_heatmap(subsample)

# energy factor reps
top_energy_factor(df_top_median, subsample)

# all
heatmaps = get_heatmaps(subsample)

