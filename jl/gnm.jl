module GNM_Mod

using LinearAlgebra
using Statistics
using MAT
using Distances
include("gnm_utils.jl")
include("graph_utils.jl")

abstract type GNM end

mutable struct GNM_Binary <: GNM
    A_Y::Matrix{Float64}
    D::Matrix{Float64}
    A_init::Matrix{Float64}
    params::Matrix{Float64}
    i_model::Int

    A_current::Matrix{Float64}
    K_current::Matrix{Float64}
    k_current::Vector{Float64}
    stat::Vector{Float64}

    m::Int
    m_seed::Int
    n_nodes::Int
    u::Vector{Int}
    v::Vector{Int}
    b::Matrix{Int}
    K::Matrix{Float64}
    energy_Y::Matrix{Float64}
    epsilon::Float64
end

mutable struct GNM_Weighted <: GNM
    A_Y::Matrix{Float64}
    D::Matrix{Float64}
    A_init::Matrix{Float64}
    params::Matrix{Float64}
    i_model::Int

    A_current::Matrix{Float64}
    K_current::Matrix{Float64}
    k_current::Vector{Float64}
    stat::Vector{Float64}

    m::Int
    m_seed::Int
    n_nodes::Int
    u::Vector{Int}
    v::Vector{Int}
    b::Matrix{Int}
    K::Matrix{Float64}
    energy_Y::Matrix{Float64}
    epsilon::Float64
end

function GNM(
    A_Y::Matrix{Float64},
    D::Matrix{Float64},
    A_init::Matrix{Float64},
    params::Matrix{Float64},
    i_model::Int,
    weighted::Bool=false)
    """
    Outer constructor for GNM structure
    """
    epsilon = 1e-5

    # number of edges and nodes
    m = sum(A_Y) / 2
    m_seed = sum(A_init) / 2
    n_nodes = size(A_Y, 2)

    # prepare outputs
    b = zeros(Int(m), Int(size(params, 1)))
    K = zeros(Int(size(params, 1)), 4)

    # get upper tri indices, TODO: refactor
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
    energy_Y[1, :] = sum(A_Y, dims=1)
    energy_Y[2, :] = get_clustering_coeff(A_Y, n_nodes)
    energy_Y[3, :] = betweenness_centrality(A_Y, n_nodes)
    energy_Y[4, :] = sum((D .* A_Y), dims=1)

    A_current = zeros(n_nodes, n_nodes)
    K_current = zeros(n_nodes, n_nodes)
    k_current = zeros(n_nodes)
    stat = zeros(n_nodes)

    if !weighted
        return GNM_Binary(A_Y, D, A_init, params, i_model, A_current, K_current,
            k_current, stat, m, m_seed, n_nodes, u, v, b, K, energy_Y, epsilon)
    else
        #TODO
    end
end

function init_K(model::GNM)
    if model.i_model == 1 # spatial
        K = ones(model.n_nodes, model.n_nodes)

    elseif model.i_model == 2 # neighbors
        K = model.A_current * model.A_current .* .~Matrix{Bool}(I, size(model.A_current, 1), size(model.A_current, 1))

    elseif model.i_model == 3 # matching
        K = get_matching_indices(model.A_current)

    elseif (model.i_model >= 4) && (model.i_model <= 13) # clustering or degree
        if (model.i_model <= 8) # clustering
            model.stat = get_clustering_coeff(model.A_current, model.n_nodes)
        else # degree
            model.stat = dropdims(sum(model.A_current, dims=1), dims=1)
        end

        if (model.i_model == 4 || model.i_model == 9) # avg
            K = (model.stat' .+ model.stat) ./ 2
        elseif (model.i_model == 5 || model.i_model == 10) # min
            K = min.(model.stat', model.stat)
        elseif (model.i_model == 6 || model.i_model == 11) # max
            K = max.(model.stat', model.stat)
        elseif (model.i_model == 7 || model.i_model == 12) # dist
            K = abs.(model.stat' .- model.stat)
        elseif (model.i_model == 8 || model.i_model == 13) # prod
            K = (model.stat' .* model.stat)
        end
    end

    model.K_current = K .+ model.epsilon
end

function update_K(model::GNM, uu::Int, vv::Int)
    if model.i_model == 1 # spatial
        bth = [uu, vv]

    elseif model.i_model == 2 # neighbors
        bth = [uu, vv]
        bu = findall(==(1), model.A_current[uu, :])
        bv = findall(==(1), model.A_current[vv, :])
        bu = bu[bu.!=vv]
        bv = bv[bv.!=uu]
        model.K_current[vv, bu] .+= 1
        model.K_current[bu, vv] .+= 1
        model.K_current[uu, bv] .+= 1
        model.K_current[bv, uu] .+= 1

    elseif model.i_model == 3 # matching
        bth = [uu, vv]
        update_uu = findall(==(1), model.A_current * model.A_current[:, uu])
        update_vv = findall(==(1), model.A_current * model.A_current[:, vv])
        update_uu = filter(x -> x != uu, update_uu)
        update_vv = filter(x -> x != vv, update_vv)

        # TODO: Refactor
        for j in update_uu
            intersect = sum(model.A_current[uu, :] .* model.A_current[j, :])
            union = (model.k_current[uu] + model.k_current[j]) - 2 * model.A_current[uu, j]
            score = intersect > 0 ? (intersect * 2) / union : model.epsilon
            model.K_current[uu, j] = model.K_current[j, uu] = score
        end

        for j in update_vv
            intersect = sum(model.A_current[vv, :] .* model.A_current[j, :])
            union = (model.k_current[vv] + model.k_current[j]) - 2 * model.A_current[vv, j]
            score = intersect > 0 ? (intersect * 2) / union : model.epsilon
            model.K_current[vv, j] = model.K_current[j, vv] = score
        end


    elseif (model.i_model >= 4) && (model.i_model <= 13) # clustering or degree 
        if (model.i_model <= 8) # clustering
            # update the clustering coefficient at uu and vv
            bu = findall(==(1), model.A_current[uu, :])
            bv = findall(==(1), model.A_current[vv, :])
            su = Int.(model.A_current[bu, :][:, bu])
            sv = Int.(model.A_current[bv, :][:, bv])
            model.stat[uu] = model.k_current[uu] > 1 ? sum(su) / (model.k_current[uu]^2 - model.k_current[uu]) : 0
            model.stat[vv] = model.k_current[vv] > 1 ? sum(sv) / (model.k_current[vv]^2 - model.k_current[vv]) : 0

            # update the clustering coefficient at common neighbors
            bth = intersect(bu, bv)

            model.stat[bth] = model.stat[bth] .+ 2 ./ (model.k_current[bth] .^ 2 .- model.k_current[bth])
            model.stat[model.k_current.<2] .= 0
            bth = union(bth, [uu, vv])
        else # degree
            bth = [uu, vv]
            model.stat = model.k_current
        end

        if (model.i_model == 4 || model.i_model == 9)
            model.K_current[bth, :] = (model.stat' .+ model.stat[bth]) ./ 2 .+ model.epsilon
            model.K_current[:, bth] = model.K_current[bth, :]'
        elseif (model.i_model == 5 || model.i_model == 10)
            model.K_current[bth, :] = min.(model.stat', model.stat[bth]) .+ model.epsilon
            model.K_current[:, bth] = model.K_current[bth, :]'
        elseif (model.i_model == 6 || model.i_model == 11)
            model.K_current[bth, :] = max.(model.stat', model.stat[bth]) .+ model.epsilon
            model.K_current[:, bth] = model.K_current[bth, :]'
        elseif (model.i_model == 7 || model.i_model == 12)
            model.K_current[bth, :] = abs.(model.stat' .- model.stat[bth]) .+ model.epsilon
            model.K_current[:, bth] = model.K_current[bth, :]'
        elseif (model.i_model == 8 || model.i_model == 13)
            model.K_current[bth, :] = (model.stat' .* model.stat[bth]) .+ model.epsilon
            model.K_current[:, bth] = model.K_current[bth, :]'
        end
    end

    return bth
end


function generate_models(model::GNM)
    # start model generation
    for i_param in 1:size(model.params, 1)
        eta, gamma = model.params[i_param, :]
        model.A_current = copy(model.A_init)
        model.k_current = dropdims(sum(model.A_current, dims=1), dims=1)
        init_K(model)

        # initiate probability
        Fd = model.D .^ eta
        Fk = model.K_current .^ gamma
        Ff = Fd .* Fk .* (model.A_current .== 0)
        P = [Ff[model.u[i], model.v[i]] for i in 1:length(model.u)]

        for iEdge in (model.m_seed+1):model.m
            # probabilistically select a new edge
            C = [0; cumsum(P)]
            r = sum(C .<= (rand() * C[end]))
            uu, vv = model.u[r], model.v[r]
            model.k_current[uu] += 1
            model.k_current[vv] += 1
            model.A_current[uu, vv] = model.A_current[vv, uu] = 1

            # update K
            bth = update_K(model, uu, vv)

            # update the probabilities
            for bth_i in bth
                Ff[bth_i, :] = Ff[:, bth_i] = Fd[bth_i, :] .* model.K_current[bth_i, :] .^ gamma .* (model.A_current[bth_i, :] .== 0)
            end
            P = [Ff[model.u[i], model.v[i]] for i in 1:length(model.u)]
        end

        # evaluate the param combination
        edge_indices = [Int((idx[1] + 1) * 10^(ceil(log10(idx[2] + 1)))) + idx[2] for idx in findall(==(1), model.A_current)]
        energy_Y_head = zeros(4, model.n_nodes)
        energy_Y_head[1, :] = sum(model.A_current, dims=1)
        energy_Y_head[2, :] = get_clustering_coeff(model.A_current, model.n_nodes)
        energy_Y_head[3, :] = betweenness_centrality(model.A_current, model.n_nodes)
        energy_Y_head[4, :] = sum((model.D .* model.A_current), dims=1)
        model.K[i_param, 1] = ks_test(model.energy_Y[1, :], energy_Y_head[1, :])
        model.K[i_param, 2] = ks_test(model.energy_Y[2, :], energy_Y_head[2, :])
        model.K[i_param, 3] = ks_test(model.energy_Y[3, :], energy_Y_head[3, :])
        model.K[i_param, 4] = ks_test(model.energy_Y[4, :], energy_Y_head[4, :])
    end
end
end