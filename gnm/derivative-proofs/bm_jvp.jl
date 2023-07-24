using ForwardDiff: derivative
using LinearAlgebra: Diagonal, norm, triu
using StatsBase: sample
using Random: seed!
using BenchmarkTools: @btime

include("../test_data.jl")

# load synthetic data
W_Y, D, A_init = load_weight_test_data()
A_Y = Float64.(W_Y .> 0);
α = 0.01
ω = 0.9
ϵ = 1e-5
m_seed = Int(sum(A_init))
m_all = Int(sum(A_Y))
resolution = 0.01
steps = 5

seed!(21)

zero_indices = (findall(==(1), triu(abs.(A_init .- 1), 1)))
edges_to_add = sample(zero_indices, m_all - m_seed; replace=false);

function norm_obj_func_auto_diff(W, D, ω)
    # compute S
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))

    # compute the objective
    return sum((exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric()) .* D) .^ ω)
end;

function run(
    A_init::Matrix{Float64},
    m_max::Int)
    A_current = copy(A_init)
    W_current = copy(A_init)

    for m in 1:m_max
        # Get the edge, order of added edges is fixed
        edge_idx = edges_to_add[m-m_seed]
        rev_idx = CartesianIndex(edge_idx[2], edge_idx[1])
        A_current[edge_idx] = W_current[edge_idx] = 1
        A_current[rev_idx] = W_current[rev_idx] = 1
        edge_indices = findall(!=(0), triu(A_current, 1))

        # Compute the derivative
        W = copy(W_current)
        for edge in edge_indices
            # because we break symmmetry,we take differentiate with respect to both edges at once
            tangent = zeros(size(W_current))
            tangent[edge] = tangent[CartesianIndex(edge[2], edge[1])] = 1.0
            g(t) = norm_obj_func_auto_diff(W + t * tangent, D, ω)

            # update the weight matrix
            W_current[edge] = max(0, W_current[edge] - (α * derivative(g, 0.0)))
            W_current[CartesianIndex(edge[2], edge[1])] = W_current[edge]
        end
    end
    return W_current
end;



x = run(A_init, 10)
sum(x)


@btime W_res_auto = run(A_init, 10);