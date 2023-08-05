function load_files()
    # load generate
    res_files = filter(file -> endswith(file, ".res"), readdir("/store/DAMTPEGLEN/mw894/data/weighted"))
    df_all = DataFrame[]

    for (i_res_files, res_file) in enumerate(res_files)

        file = h5open(joinpath("/store/DAMTPEGLEN/mw894/data/weighted", res_file), "r")
        df_file = DataFrame()

        # read meta data
        meta_group = file["meta"]
        η = read_attribute(meta_group, "eta")
        γ = read_attribute(meta_group, "gamma")

        # read loss
        loss_tracker = read(file, "loss_tracker")
        eval_tracker = read(file, "eval_tracker") # n_iter x 3
        eval_last = eval_tracker[end, :]
        close(file)

        println("File: ", res_file, "params: ", params)

        # make df
        df_file.file_name = [file_path]
        df_file.eta = [η]
        df_file.gamma = [γ]
        df_file.loss = [loss_tracker[-1]]
        df_file.KS_S = [eval_last[1]]
        df_file.KS_WC = [eval_last[2]]
        df_file.KS_WB = [eval_last[3]]

        push!(df_all, df_file)
    end
    df_all = vcat(df_all...)
    df_all.KS_W_MAX = map(maximum, eachrow(df_all[!, ["KS_S", "KS_WC", "KS_WB"]]))

    return df_all
end

function landscape_plot(df)
    # best KS for eta gamma
    df_eta_gamma = combine(groupby(df, [:eta, :gamma]), :KS_MAX => minimum => :KS_MAX, :KS_S => minimum => :KS_S, :KS_WC => minimum => :KS_WC, :KS_WB => minimum => :KS_WB)
    sort!(df_eta_gamma, [:eta, :gamma])

    landscape = reshape(df_eta_gamma.KS_MAX, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p1 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="Best: KS Max", clim=(0, 1), c=:viridis)

    landscape = reshape(df_eta_gamma.KS_S, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p2 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="Best: KS S", clim=(0, 1), c=:viridis)

    landscape = reshape(df_eta_gamma.KS_WC, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p3 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="Best: KS KC", clim=(0, 1), c=:viridis)

    landscape = reshape(df_eta_gamma.KS_WB, (length(unique(df_eta_gamma.eta)), length(unique(df_eta_gamma.gamma))))
    p4 = heatmap(unique(df_eta_gamma.eta), unique(df_eta_gamma.gamma), landscape, xlabel="eta", ylabel="gamma", title="Best: KS WB", clim=(0, 1), c=:viridis)

    p = plot(p1, p2, p3, p4;
        layout=grid(2, 2),
        fmt=:pdf,
        size=(1500, 1400),
        margin=8mm)

    savefig(p, "/home/mw894/diss/gnm/analysis-weighted/w_landscapes.pdf")
end