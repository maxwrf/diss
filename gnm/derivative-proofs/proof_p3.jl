"""
This file demonstrates that using the jacobian from ForwardDiff.jl that with 
the normalization step we can separate the two derivatives using the chain rule.

Author: Max Würfel
Date: July 18, 2023
"""

using LinearAlgebra: Diagonal, I, norm, lu
using ExponentialUtilities
using ForwardDiff: gradient, derivative, jacobian

W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]

# Ground truth using gradient
function f(W)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    norm_fact = sqrt.(node_strengths * node_strengths')
    return sum(exponential!(copyto!(similar(W), W ./ norm_fact), ExpMethodGeneric()))
end

g_result = gradient(f, W);
display(g_result)


# Jacobian proof functions
function func_gx(W)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))
    return S * W * S
end

function func_fg(W)
    return exponential!(copyto!(similar(W), W), ExpMethodGeneric())
end

function forward_diff_j(f::Function, W::Matrix{Float64})
    return jacobian(f, W)
end

node_strengths = dropdims(sum(W, dims=2), dims=2)
node_strengths[node_strengths.==0] .= 1e-5
S = sqrt(inv(Diagonal(node_strengths)))

dgx = forward_diff_j(func_gx, W)
dfg = forward_diff_j(func_fg, S * W * S)
chain_result = reshape(sum(dfg' * dgx, dims=1), 3, 3)
display(chain_result)





