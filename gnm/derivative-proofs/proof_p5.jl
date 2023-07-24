using ForwardDiff: derivative
using LinearAlgebra: Diagonal, norm, triu
using StatsBase: sample
using Random: seed!
using ExponentialUtilities


# load synthetic data


function norm_obj_func_auto_diff(W, D, ω)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))
    return sum((exponential!(copyto!(similar(W), S * W * S), ExpMethodHigham2005()) .* D) .^ ω)
end;

W = [1 0.2; 0.2 0]
D = [1 3; 2 4]
ω = 0.9

tangent = zeros(size(W))
tangent[1, 2] = 1.0
g(t) = norm_obj_func_auto_diff(W + t * tangent, D, ω)
derivative(g, 0.0)