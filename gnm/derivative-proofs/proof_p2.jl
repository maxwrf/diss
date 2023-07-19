"""
This file demonstrates that using the jacobian vector product as part of the
forward differentiation, it is possible to achieve the same result as using
the tangent approximation or ForwardDiff.jl's gradient function, but faster.

Author: Max Würfel
Date: July 18, 2023
"""

using LinearAlgebra: Diagonal, I, norm, lu
using ExponentialUtilities
using ForwardDiff: gradient, derivative, jacobian
using Polynomials: fit

W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]


function f(W)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    norm_fact = sqrt.(node_strengths * node_strengths')
    return sum(exponential!(copyto!(similar(W), W ./ norm_fact), ExpMethodGeneric()))
end


g_result = gradient(f, W);
display(g_result)


function forward_diff_jvp(W::Matrix{Float64}, tangent)
    function f(W)
        node_strengths = dropdims(sum(W, dims=2), dims=2)
        node_strengths[node_strengths.==0] .= 1e-5
        S = sqrt(inv(Diagonal(node_strengths)))
        return sum(exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric()))
    end

    g(t) = f(W + t * tangent)
    JVP = derivative(g, 0.0)
    return JVP
end

function jvp_all_edges(W::Matrix{Float64})
    result = zeros(size(W))
    for edge in collect(CartesianIndices(W))
        tangent = zeros(size(W))
        tangent[edge] = 1.0
        result[edge] = forward_diff_jvp(W, tangent)
    end
    return result
end

jvp_results = jvp_all_edges(W)
display(jvp_results)

# side note
function forward_diff_j(W::Matrix{Float64})
    # Column indices for retrieval
    indices = collect(CartesianIndices(W))
    index_vec = sort(vec(indices), by=x -> x[1])

    function f(W)
        node_strengths = dropdims(sum(W, dims=2), dims=2)
        node_strengths[node_strengths.==0] .= 1e-5
        S = sqrt(inv(Diagonal(node_strengths)))
        return exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric())
    end
    J = jacobian(f, W)


    return J
end

forward_diff_j(W)
