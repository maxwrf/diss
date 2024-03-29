using LinearAlgebra
using ExponentialUtilities
using Zygote
using Plots
using Measures
using StatsBase
using HDF5
using Random


Random.seed!(1234)

include("gnm/graph_utils.jl")
include("gnm/test_data.jl")
include("gnm/gnm_utils.jl")

function dist_plot(W_Y, dist_tracker)
    dists_Y = [
        dropdims(sum(weight_conversion(W_Y), dims=1), dims=1),
        dropdims(clustering_coef_wu(weight_conversion(W_Y)), dims=2),
        betweenness_wei(weight_conversion(W_Y))
    ]

    # pred
    dists_Y_pred = [dist_tracker[end, 1, :], dist_tracker[end, 2, :], dist_tracker[end, 3, :]]

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

    savefig(p, "w_dist.pdf")
end

function learning_plot(loss_tracker, eval_tracker, η, γ)
    evals = hcat(eval_tracker...)'
    evals = eval_tracker
    p = plot(evals, lw=2, marker=3, label=["Strength" "Clustering" "Betweenness"], margin=5mm, fmt=:pdf)
    xlabel!("Iteration")
    ylabel!("KS statisitc")
    title!("η =" * string(η) * ", γ =" * string(γ))
    plot!(twinx(), loss_tracker, color=:black, marker=3, linestyle=:dash, legend=false, ylabel="Loss")
    savefig(p, "learning_curve.pdf")
end

function callback(x, W_Y, f_loss, eval_tracker, loss_tracker, dist_tracker)
    W_head = Matrix(Symmetric(x))

    energy_W_head = zeros(3, size(W_head, 1))
    energy_W_head[1, :] = sum(weight_conversion(W_head), dims=1)
    energy_W_head[2, :] = clustering_coef_wu(weight_conversion(W_head))
    energy_W_head[3, :] = betweenness_wei(weight_conversion(W_head))

    energy_W = zeros(3, size(W_Y, 1))
    energy_W[1, :] = sum(weight_conversion(W_Y), dims=1)
    energy_W[2, :] = clustering_coef_wu(weight_conversion(W_Y))
    energy_W[3, :] = betweenness_wei(weight_conversion(W_Y))

    K_W = zeros(3)
    K_W[1] = ks_test(energy_W[1, :], energy_W_head[1, :])
    K_W[2] = ks_test(energy_W[2, :], energy_W_head[2, :])
    K_W[3] = ks_test(energy_W[3, :], energy_W_head[3, :])

    cur_loss = f_loss(x)
    push!(eval_tracker, K_W)
    push!(loss_tracker, cur_loss)
    push!(dist_tracker, energy_W_head)

    println("loss: ", cur_loss, ", K_W: ", K_W)
end

function gradient_descent(x, D, W_Y, η, γ, T, callback, eval_tracker, loss_tracker, dist_tracker)
    function f_loss(W)
        W = Matrix(Symmetric(W))
        # compute S
        node_strengths = dropdims(sum(W, dims=2), dims=2)
        tempnode_strengths = Zygote.Buffer(node_strengths)
        tempnode_strengths[:] = node_strengths
        for (i, node_strength) in enumerate(node_strengths)
            if node_strength == 0
                tempnode_strengths[i] = 1e-5
            end
        end
        node_strengths = copy(tempnode_strengths)
        S = Matrix(sqrt(inv(Diagonal(node_strengths))))

        # compute communicability, diagonal is zero
        C = exp(S * W * S) .* (Matrix(I(size(W, 1))) .!= 1)

        optimal_eficiency = global_efficiency((1 / maximum(W_Y) .+ zeros(size(W_Y))) .* (W_Y .> 0) .* D)
        current_efficiency = global_efficiency(W .* D)
        communicability = sum((W .* C .* D))

        return (optimal_eficiency - current_efficiency) + γ * communicability
    end

    for t in 1:T
        ∇f = gradient(f_loss, x)[1] # compute the gradient
        ∇f = replace(∇f, NaN => 0.0)
        x[∇f.!=0] = clamp!((x.-η*∇f)[∇f.!=0], 1e-5, Inf) # update parameters in direction of -∇f
        callback(x, W_Y, f_loss, eval_tracker, loss_tracker, dist_tracker) # the callback will be used to save intermediate values
    end
    x
end

function main_test()
    W_Y, D, _, _ = load_weight_test_data()
    η = 0.001
    γ = 1
    n_iter = 30

    η = 0.002
    γ = 0.844444

    loss_tracker = Float64[]
    eval_tracker = Vector[]
    dist_tracker = Matrix{Float64}[]

    W_init = rand(size(W_Y)...)
    W_init = W_init .* (W_Y .> 0)
    W_init = Matrix(Symmetric(W_init))

    x = gradient_descent(triu(W_init), D, W_Y, η, γ, n_iter, callback, eval_tracker, loss_tracker, dist_tracker)
    dist_plot(W_Y, dist_tracker)
    learning_plot(loss_tracker, eval_tracker, η, γ)

end


function main()
    n_iter = 20
    W_Y, D, _, _ = load_weight_test_data()
    W_init = rand(size(W_Y)...)
    W_init = W_init .* (W_Y .> 0)
    W_init = Matrix(Symmetric(W_init))

    # params
    n_runs = 100
    eta_range = [0.001, 0.01] # [0.8, 1.2] omeg
    gamma_range = [0.8, 1.2]
    params = generate_param_space(n_runs, eta_range, gamma_range)

    # write files
    for n_combi in 1:size(params, 1)
        η = params[n_combi, 1]
        γ = params[n_combi, 2]

        loss_tracker = Float64[]
        eval_tracker = Vector[]
        dist_tracker = Matrix{Float64}[]
        @time W_final = gradient_descent(triu(W_init), D, W_Y, η, γ, n_iter, callback, eval_tracker, loss_tracker, dist_tracker)

        # save the results
        file = h5open("/Users/maxwuerfek/code/diss/data/" * string(n_combi) * ".res", "w")

        write(file, "W_final", W_final)
        write(file, "loss_tracker", loss_tracker)
        write(file, "eval_tracker", Matrix(hcat(eval_tracker...)')) # n_iter x 3

        # niter x 3 x n_nodes
        distr_tracker_store = zeros(length(dist_tracker), size(dist_tracker[1])...)
        for (i, dist) in enumerate(dist_tracker)
            distr_tracker_store[i, :, :] = dist
        end
        write(file, "dist_tracker", distr_tracker_store)

        meta_group = create_group(file, "meta")
        attributes(meta_group)["eta"] = η
        attributes(meta_group)["gamma"] = γ
        close(file)
    end
end

main()