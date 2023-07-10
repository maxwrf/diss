using LinearAlgebra
using Statistics
using MAT
using Distances
include("gnm_utils.jl")
include("test_data.jl")
include("graph_utils.jl")

function init_K(modelIdx::Int, nNodes::Int)
    if modelIdx == 1 # spatial
        return ones(nNodes, nNodes)
    end
end

function update_K(K_current, modelIdx::Int)
    if modelIdx == 1 # spatial
        return K_current
    end
    return K_current
end


function generate_models(A, D, A_init, params, iModel)
    # number of edges and nodes
    m = sum(A) / 2
    m_seed = sum(A_init) / 2
    n_nodes = size(A, 2)

    # prepare outputs
    b = zeros(Int(m), Int(size(params, 1)))
    K = zeros(Int(size(params, 1)), 4)

    # get upper tri indices
    u = Int[]
    v = Int[]

    for i in 1:n_nodes
        for j in (i+1):n_nodes
            push!(u, i)
            push!(v, j)
        end
    end

    # compute sample energy
    energy_Y = zeros(4, n_nodes)
    energy_Y[1, :] = sum(A, dims=1)
    energy_Y[2, :] = get_clustering_coeff(A, n_nodes)
    energy_Y[3, :] = betweenness_centrality(A, n_nodes)
    energy_Y[4, :] = sum((D .* A), dims=1)

    # start model generation
    for iParam in 1:size(params, 1)
        eta, gamma = params[iParam, :]
        A_current = copy(A_init)
        k_current = sum(A_current, dims=1)
        K_current = init_K(iModel, n_nodes)

        # initiate probability
        Fd = D .^ eta
        Fk = K_current .^ gamma
        Ff = Fd .* Fk .* (A_current .== 0)
        P = [Ff[u[i], v[i]] for i in 1:length(u)]

        for iEdge in (m_seed+1):m
            # probabilitically select a new edge
            C = [0; cumsum(P)]
            r = sum(C .<= (rand() * C[end]))
            uu, vv = u[r], v[r]
            k_current[uu] += 1
            k_current[vv] += 1
            A_current[uu, vv] = A_current[vv, uu] = 1
            bth = [uu, vv]

            # update K
            K_current = update_K(K_current, iModel)

            # update the probabilities
            for bth_i in bth
                Ff[bth_i, :] = Ff[:, bth_i] = Fd[:, bth_i] .* K_current[bth_i, :] .^ gamma .* (A_current[bth_i, :] .== 0)
            end
            P = [Ff[u[i], v[i]] for i in 1:length(u)]
        end

        # evaluate the param combination
        edge_indices = [Int((idx[1] + 1) * 10^(ceil(log10(idx[2] + 1)))) + idx[2] for idx in findall(==(1), A_current)]
        energy_Y_head = zeros(4, n_nodes)
        energy_Y_head[1, :] = sum(A_current, dims=1)
        energy_Y_head[2, :] = get_clustering_coeff(A_current, n_nodes)
        energy_Y_head[3, :] = betweenness_centrality(A_current, n_nodes)
        energy_Y_head[4, :] = sum((D .* A_current), dims=1)
        K[iParam, 1] = ks_test(energy_Y[1, :], energy_Y_head[1, :])
        K[iParam, 2] = ks_test(energy_Y[2, :], energy_Y_head[2, :])
        K[iParam, 3] = ks_test(energy_Y[3, :], energy_Y_head[3, :])
        K[iParam, 4] = ks_test(energy_Y[4, :], energy_Y_head[4, :])
    end

    return K
end
