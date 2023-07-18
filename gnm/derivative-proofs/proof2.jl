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

jvp_results = zeros(size(W))
for edge in collect(CartesianIndices(W))
    tangent = zeros(size(W))
    tangent[edge] = 1.0
    jvp_results[edge] = forward_diff_jvp(W, tangent)
end



# side note
function forward_diff_j(W::Matrix{Float64})::Vector{Float64}
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

    results = zeros(length(index_vec))
    display(J)
    tangent = vec(permutedims(exp(W), [2, 1]))
    for (i_edge, edge) in enumerate(index_vec)
        # we get all partial derivative positions that are non-zero
        Jₓ = J[:, findfirst(x -> x == edge, index_vec)]
        results[i_edge] = sum(Jₓ)
    end

    return results
end

forward_diff_j(W)
