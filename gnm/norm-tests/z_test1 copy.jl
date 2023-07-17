using Polynomials
using ForwardDiff
using ExponentialUtilities
using LinearAlgebra
using BenchmarkTools
using StatsBase: sample
using Random
include("test_data.jl")
Random.seed!(123)

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
            sum_comm[i_rep] = f(W_copy, edge_idx)
        end

        results[i_edge] = fit(reps, sum_comm, 1)[1]
    end

    return results
end

function test_tangent(m_max::Int, normalize::Bool)
    A_current = copy(A_init)
    W_current = copy(A_init)

    if normalize
        function f(W, edge_idx)
            node_strengths = dropdims(sum(W, dims=2), dims=2)
            node_strengths[node_strengths.==0] .= 1e-5
            norm_fact = sqrt.(node_strengths .* node_strengths')
            temp = W .* norm_fact
            return (sum(exp(temp)))
        end
    else
        f = (W, edge_idx) -> (sum(exp(W)) * D[edge_idx])^ω
    end



    for m in 1:m_max
        # Get the edge
        edge_idx = edges_to_add[m-m_seed]
        rev_idx = CartesianIndex(edge_idx[2], edge_idx[1])
        A_current[edge_idx] = W_current[edge_idx] = 1
        A_current[rev_idx] = W_current[rev_idx] = 1
        edge_indices = findall(!=(0), triu(A_current, 1))

        tangent_d = tangent_approx(f, W_current, edge_indices, resolution, steps)
        println(round.(tangent_d, digits=10))

        for (i_edge, edge) in enumerate(edge_indices)
            W_current[edge] -= (α * tangent_d[i_edge])
            W_current[edge] = max(0, W_current[edge])
            W_current[CartesianIndex(edge[2], edge[1])] = W_current[edge]
        end
    end
    return W_current
end;

function forward_diff_jvp(
    W::Matrix{Float64},
    edges::Vector{CartesianIndex{2}}
)
    tangent = ones(size(W))

    function diff_exp(W)
        node_strengths = dropdims(sum(W, dims=2), dims=2)
        node_strengths[node_strengths.==0] .= 1e-5
        # is approx if cloese to zero not == 0 
        norm_fact = sqrt.(node_strengths .* node_strengths')
        temp = W .* norm_fact
        return exponential!(copyto!(similar(temp), temp), ExpMethodGeneric())
    end

    g(t) = diff_exp(W + t * tangent)
    JVP = ForwardDiff.derivative(g, 0.0)
    return JVP[edges]
end


function test(m_max::Int, normalize::Bool)
    A_current = copy(A_init)
    W_current = copy(A_init)

    for m in 1:m_max
        # Get the edge
        edge_idx = edges_to_add[m-m_seed]
        rev_idx = CartesianIndex(edge_idx[2], edge_idx[1])
        A_current[edge_idx] = W_current[edge_idx] = 1
        A_current[rev_idx] = W_current[rev_idx] = 1
        edge_indices = findall(!=(0), triu(A_current, 1))

        derivate = forward_diff_jvp(W_current, edge_indices)

        display(round.(derivate, digits=5))

        for (i_edge, edge) in enumerate(edge_indices)
            W_current[edge] -= (α * derivate[i_edge])
            W_current[edge] = max(0, W_current[edge])
            W_current[CartesianIndex(edge[2], edge[1])] = W_current[edge]
        end

    end
    return W_current
end;



# load synthetic data
W_Y, D, A_init = load_weight_test_data()


W_Y = W_Y[10:15, 10:15]
D = D[10:15, 10:15]
A_init = A_init[10:15, 10:15]

A_Y = Float64.(W_Y .> 0);

α = 0.01
ω = 0.9
ϵ = 1e-5
m_seed = Int(sum(A_init))
m_all = Int(sum(A_Y))
resolution = 0.01
steps = 5




zero_indices = (findall(==(1), triu(abs.(A_init .- 1), 1)))
edges_to_add = sample(zero_indices, m_all - m_seed; replace=false);

display(edges_to_add)

@time W_res_frechet = test(3, true);
@time W_res_tangent = test_tangent(3, true);