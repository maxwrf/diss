using LinearAlgebra: Diagonal
using ExponentialUtilities
using ForwardDiff: gradient
using Polynomials: fit

W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]

# Showing these are equvalent
# node_strengths = dropdims(sum(W, dims=2), dims=2)
# node_strengths[node_strengths.==0] .= 1e-5
# S = sqrt(inv(Diagonal(node_strengths)))
# norm_fact = sqrt.(node_strengths * node_strengths')
# display(W ./ norm_fact)
# display(S * W * S)

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




function frechet_algo(A::Matrix{Float64}, E::Matrix{Float64})

    n = size(A, 1)
    s = nothing
    ident = Matrix{Float64}(I, n, n)
    A_norm_1 = norm(A, 1)
    m_pade_pairs = [(3, _diff_pade3), (5, _diff_pade5), (7, _diff_pade7), (9, _diff_pade9)]
    for (m, pade) in m_pade_pairs
        if A_norm_1 <= ell_table_61[m]
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break
        end
    end
    if s == nothing
        # scaling
        s = max(0, ceil(Int, log2(A_norm_1 / ell_table_61[13])))
        A *= 2.0^-s
        E *= 2.0^-s
        U, V, Lu, Lv = _diff_pade13(A, E, ident)
    end

    # factor once and solve twice
    lu_piv = lu(-U + V)
    R = lu_piv \ (U + V)
    L = lu_piv \ (Lu + Lv + ((Lu - Lv) * R))

    # repeated squaring
    for k in 1:s
        L = R * L + L * R
        R = R * R
    end
    return R, L
end



t_result = tangent_approx(f, W, vec(collect(CartesianIndices(W))), 0.01);
g_result = gradient(f, W);

display(t_result)
display(g_result)

#TODO: Frechet
# W = [1 0.2
#     0.2 0]


# node_strengths = dropdims(sum(W, dims=2), dims=2)
# node_strengths[node_strengths.==0] .= 1e-5
# S = sqrt(inv(Diagonal(node_strengths)))
# R, L = frechet_algo(S * W * S, ones(size(W)))
# display(R)
# display(L)

# function f(W)
#     node_strengths = dropdims(sum(W, dims=2), dims=2)
#     node_strengths[node_strengths.==0] .= 1e-5
#     S = sqrt(inv(Diagonal(node_strengths)))
#     return sum(S * W * S)
# end
# X = ForwardDiff.gradient(f, W)