using LinearAlgebra
using Statistics
using MAT
using Distances
include("gnm_utils.jl")
include("graph_utils.jl")

function init_K(A_current, modelIdx::Int, n_nodes::Int)
    epsilon = 1e-5
    stat = nothing

    if modelIdx == 1 # spatial
        K = ones(n_nodes, n_nodes)

    elseif modelIdx == 2 # neighbors
        K = A_current * A_current .* .~Matrix{Bool}(I, size(A_current, 1), size(A_current, 1))

    elseif modelIdx == 3 # matching
        K = get_matching_indices(A_current)

    elseif (modelIdx >= 4) && (modelIdx <= 13) # clustering or degree
        if (modelIdx <= 8) # clustering
            stat = get_clustering_coeff(A_current, n_nodes)
        else # degree
            stat = dropdims(sum(A_current, dims=1), dims=1)
        end

        if (modelIdx == 4 || modelIdx == 9) # avg
            K = (stat' .+ stat) ./ 2
        elseif (modelIdx == 5 || modelIdx == 10) # min
            K = min.(stat', stat)
        elseif (modelIdx == 6 || modelIdx == 11) # max
            K = max.(stat', stat)
        elseif (modelIdx == 7 || modelIdx == 12) # dist
            K = abs.(stat' .- stat)
        elseif (modelIdx == 8 || modelIdx == 13) # prod
            K = (stat' .* stat)
        end
    end

    return K .+ epsilon, stat
end

function update_K(A_current, K_current, k_current, modelIdx::Int, uu::Int, vv::Int, stat)
    epsilon = 1e-5

    if modelIdx == 1 # spatial
        bth = [uu, vv]

    elseif modelIdx == 2 # neighbors
        bth = [uu, vv]
        bu = findall(==(1), A_current[uu, :])
        bv = findall(==(1), A_current[vv, :])
        bu = bu[bu.!=vv]
        bv = bv[bv.!=uu]
        K_current[vv, bu] .+= 1
        K_current[bu, vv] .+= 1
        K_current[uu, bv] .+= 1
        K_current[bv, uu] .+= 1

    elseif modelIdx == 3 # matching
        bth = [uu, vv]
        update_uu = findall(==(1), A_current * A_current[:, uu])
        update_vv = findall(==(1), A_current * A_current[:, vv])
        update_uu = filter(x -> x != uu, update_uu)
        update_vv = filter(x -> x != vv, update_vv)

        # TODO: Refactor
        for j in update_uu
            intersect = sum(A_current[uu, :] .* A_current[j, :])
            union = (k_current[uu] + k_current[j]) - 2 * A_current[uu, j]
            score = intersect > 0 ? (intersect * 2) / union : epsilon
            K_current[uu, j] = K_current[j, uu] = score
        end

        for j in update_vv
            intersect = sum(A_current[vv, :] .* A_current[j, :])
            union = (k_current[vv] + k_current[j]) - 2 * A_current[vv, j]
            score = intersect > 0 ? (intersect * 2) / union : epsilon
            K_current[vv, j] = K_current[j, vv] = score
        end


    elseif (modelIdx >= 4) && (modelIdx <= 13) # clustering or degree 
        if (modelIdx <= 8) # clustering
            # update the clustering coefficient at uu and vv
            bu = findall(==(1), A_current[uu, :])
            bv = findall(==(1), A_current[vv, :])
            su = Int.(A_current[bu, :][:, bu])
            sv = Int.(A_current[bv, :][:, bv])
            stat[uu] = k_current[uu] > 1 ? sum(su) / (k_current[uu]^2 - k_current[uu]) : 0
            stat[vv] = k_current[vv] > 1 ? sum(sv) / (k_current[vv]^2 - k_current[vv]) : 0

            # update the clustering coefficient at common neighbors
            bth = intersect(bu, bv)

            stat[bth] = stat[bth] .+ 2 ./ (k_current[bth] .^ 2 .- k_current[bth])
            stat[k_current.<2] .= 0
            bth = union(bth, [uu, vv])
        else # degree
            bth = [uu, vv]
            stat = k_current
        end

        if (modelIdx == 4 || modelIdx == 9)
            K_current[bth, :] = (stat' .+ stat[bth]) ./ 2 .+ epsilon
            K_current[:, bth] = K_current[bth, :]'
        elseif (modelIdx == 5 || modelIdx == 10)
            K_current[bth, :] = min.(stat', stat[bth]) .+ epsilon
            K_current[:, bth] = K_current[bth, :]'
        elseif (modelIdx == 6 || modelIdx == 11)
            K_current[bth, :] = max.(stat', stat[bth]) .+ epsilon
            K_current[:, bth] = K_current[bth, :]'
        elseif (modelIdx == 7 || modelIdx == 12)
            K_current[bth, :] = abs.(stat' .- stat[bth]) .+ epsilon
            K_current[:, bth] = K_current[bth, :]'
        elseif (modelIdx == 8 || modelIdx == 13)
            K_current[bth, :] = (stat' .* stat[bth]) .+ epsilon
            K_current[:, bth] = K_current[bth, :]'
        end
    end

    return K_current, stat, bth
end


function generate_models(A, D, A_init, params, i_model)
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
        k_current = dropdims(sum(A_current, dims=1), dims=1)
        K_current, stat = init_K(A_current, i_model, n_nodes)

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

            # update K
            K_current, stat, bth = update_K(
                A_current, K_current, k_current, i_model, uu, vv, stat)

            # update the probabilities
            for bth_i in bth
                Ff[bth_i, :] = Ff[:, bth_i] = Fd[bth_i, :] .* K_current[bth_i, :] .^ gamma .* (A_current[bth_i, :] .== 0)
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
