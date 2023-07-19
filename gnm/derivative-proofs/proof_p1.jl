"""
This file demonstrates the Forward differentation gradient in Julia can produce
the same result as the tangent approximation of the normalized communicability
objective function.

Author: Max Würfel
Date: July 18, 2023
"""

using LinearAlgebra: Diagonal
using ExponentialUtilities
using ForwardDiff: gradient
using Polynomials: fit

W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]

function f(W)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))
    return sum(exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric()))
end

function tangent_approx(f::Function, W::Matrix{Float64}, edges::Vector{CartesianIndex{2}},
    resolution=0.01, steps=5)::Vector{Float64}
    results = zeros(length(edges))
    rep_vec = collect(range(-steps * resolution, steps * resolution, step=resolution))

    for (i_edge, edge_idx) in enumerate(edges)
        # Points for evaluation
        edge_val = W[edge_idx]
        sign_edge = sign(edge_val) == 0 ? 1 : sign(edge_val)
        reps = [edge_val + sign_edge * (max(abs(edge_val), 1e-3) * i) for i in rep_vec]


        # For each nudge save difference in communicability 
        sum_comm = zeros(length(reps))
        for (i_rep, rep) in enumerate(reps)
            W_copy = copy(W)
            W_copy[edge_idx] = rep
            sum_comm[i_rep] = f(W_copy)
        end

        results[i_edge] = fit(reps, sum_comm, 1)[1]
    end

    return results
end

t_result = tangent_approx(f, W, vec(collect(CartesianIndices(W))), 0.01);
g_result = gradient(f, W);

display(t_result)
display(g_result)
