using LinearAlgebra
using ExponentialUtilities
using Zygote
using Plots
using Measures

include("gnm/graph_utils.jl")
include("gnm/test_data.jl")
include("gnm/gnm_utils.jl")

function learning_plot(loss_tracker, eval_tracker)
    evals = hcat(eval_tracker...)'
    plot(evals, lw=2, marker=3, label=["Strength" "Clustering" "Betweenness"], margin=5mm)
    xlabel!("iteration")
    ylabel!("KS statisitc")
    title!("η =" * string(η) * ", γ =" * string(γ))
    plot!(twinx(), loss_tracker, color=:black, marker=3, linestyle=:dash, legend=false)
    ylabel!(twinx(), "loss")
end

function callback(x, eval_tracker, loss_tracker)
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

    cur_loss = loss(x)
    push!(eval_tracker, K_W)
    push!(loss_tracker, cur_loss)

    println("loss: ", cur_loss)
    println(K_W)
end

function gradient_descent(x, D, η, γ, T, callback, eval_tracker, loss_tracker)
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
        callback(x, eval_tracker, loss_tracker) # the callback will be used to save intermediate values
    end
    x
end

function main()
    W_Y, D, _, _ = load_weight_test_data()
    η = 0.001
    γ = 1
    n_iter = 30

    loss_tracker = Float64[]
    eval_tracker = Vector[]

    W_init = rand(size(W_Y)...)
    W_init = W_init .* (W_Y .> 0)
    W_init = Matrix(Symmetric(W_init))

    x = gradient_descent(triu(W_init), D, η, γ, n_iter, callback, eval_tracker, loss_tracker)
    learning_plot(loss_tracker, eval_tracker)
end

main()

