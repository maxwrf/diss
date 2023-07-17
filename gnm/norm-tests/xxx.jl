W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]
demo_edge = [CartesianIndex(1, 2)]

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

# Showing these are equvalent
node_strengths = dropdims(sum(W, dims=2), dims=2)
node_strengths[node_strengths.==0] .= 1e-5
S = sqrt(inv(Diagonal(node_strengths)))
norm_fact = sqrt.(node_strengths * node_strengths')
display(W ./ norm_fact)
display(S * W * S)

function f(W, _)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))
    return sum(exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric()))
end

tangent_approx(f, W, vec(collect(CartesianIndices(W))), 0.01)

function forward_diff_jvp(W::Matrix{Float64})
    tangent = ones(size(W))
    function f(W)
        node_strengths = dropdims(sum(W, dims=2), dims=2)
        #node_strengths[node_strengths.==0] .= 1e-5
        S = sqrt(inv(Diagonal(node_strengths)))
        return exponential!(copyto!(similar(W), S * W * S), ExpMethodGeneric())
    end

    g(t) = f(W + t * tangent)
    JVP = ForwardDiff.derivative(g, 0.0)
    return JVP
end

forward_diff_jvp(W)

ForwardDiff.gradient(f, W);

forward_diff_j(W, demo_edge)



