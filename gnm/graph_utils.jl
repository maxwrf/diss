using LinearAlgebra: diagind, I, diag

function get_matching_indices(A::Matrix{Float64})
    """
    For any two nodes in the adjacency matrix, computes the overlap in the
    connections.
    Note that if two nodes have a common neighbor, the two edges are both
    counted as part of the intersection set.
    Any connection between the two nodes is excluded.
    """
    intersect = A * A .* .~Matrix{Bool}(I, size(A, 1), size(A, 1))
    degree = dropdims(sum(A, dims=1), dims=1)
    union = (degree .+ degree') .- (2 .* A)
    m_indices = (intersect .* 2) ./ union
    replace!(m_indices, NaN => 0)
    return m_indices
end

function get_clustering_coeff(A::Matrix{Float64}, n_nodes::Int)::Vector{Float64}
    """
    Computes the clustering coefficients for each node
    A = adjacency matrix
    n = number of nodes
    """
    clu_coeff = zeros(Float64, n_nodes)
    for i_node in 1:n_nodes
        nbrs = findall(==(1), A[i_node, :])
        n_nbrs = length(nbrs)
        if n_nbrs > 1
            S = A[nbrs, :][:, nbrs]
            clu_coeff[i_node] = sum(S) / (n_nbrs^2 - n_nbrs)
        end
    end

    return clu_coeff
end


function betweenness_centrality(A::Matrix{Float64}, n::Int)::Vector{Float64}
    """
    Computes the betweenes centrality for each node
    A = adjacency matrix
    n = number of nodes
    Ref: https://arxiv.org/pdf/0809.1906.pdf
    """

    # FORWARD PASS
    d = 1  # path length
    NPd = copy(A)  # number of paths of length |d|
    NSPd = copy(A)  # number of shortest paths of length |d|
    NSP = copy(A)  # number of shortest paths of any length
    L = copy(A)

    NSP[diagind(NSP)] .= 1
    L[diagind(L)] .= 1

    while any(NSPd .== 1)
        d += 1

        # Computes the number of paths connecting i & j of length d
        NPd = NPd * A

        # if no connection between edges yet, this is the shortest path
        NSPd = NPd .* (L .== 0)

        # Add the new shortest path entries (in L multiply with length)
        NSP += NSPd
        L .+= d .* (NSPd .!= 0)
    end

    L[L.==0] .= Inf  # L for disconnected vertices is Inf
    NSP[NSP.==0] .= 1  # NSP for disconnected vertices is 1
    L[diagind(L)] .= 0  # no loops

    # BACKWARD PASS
    # dependency number of shortest paths from i to any other vertex that
    # pass through j
    DP = zeros(Float64, n, n)  # vertex on vertex dependency
    diam = d - 1  # the maximum distance between any two nodes

    # iterate from longest shortest path to shortest
    for d in diam:-1:2
        # DPd1 is dependency, shortest paths from i to any other vertex that
        # pass through i with a length of d
        DPd1 = (((L .== d) .* (1 .+ DP) ./ NSP) * A') .* ((L .== (d - 1)) .* NSP)
        DP += DPd1
    end

    return dropdims(sum(DP, dims=1) ./ 2, dims=1)
end

function weight_conversion(A::Matrix{Float64})
    A = A ./ maximum(abs.(A))
end

function clustering_coef_wu(A::Matrix{Float64})
    # Calculate the degree of each node (number of non-zero Aeights)
    K = Float64.(sum(A .!= 0, dims=2))

    # Calculate the number of 3-cycles for each node
    cyc3 = diag((A .^ (1 / 3))^3)

    # Set the clustering coefficient to 0 for nodes without 3-cycles
    K[cyc3.==0] .= Inf

    # Calculate the clustering coefficient for each node
    C = cyc3 ./ (K .* (K .- 1))

    return C
end

function betweenness_wei(G)
    # n is the number of nodes in the graph
    n = size(G, 1)

    BC = zeros(n)      # Initialize vertex betweenness

    for u in 1:n
        D = fill(Inf, n)    # Distance from u (initialized to Inf)
        D[u] = 0            # Distance from u to itself is 0
        NP = zeros(Int, n)  # Number of paths from u (initialized to 0)
        NP[u] = 1           # Number of paths from u to itself is 1
        S = ones(Bool, n)   # Distance permanence (initialized to true)
        P = zeros(Bool, n, n)   # Predecessors (initialized to false)
        Q = zeros(Int, n)   # Queue for nodes to visit (initialized to 0)
        q = n               # Order of non-increasing distance

        G1 = copy(G)
        V = [u]

        while true
            S[V] .= false    # Distance u->V is now permanent
            G1[:, V] .= 0    # No in-edges as already shortest

            for v in V
                Q[q] = v
                q -= 1
                W = findall(x -> x != 0, G1[v, :])   # Neighbours of v
                for w in W
                    Duw = D[v] + G1[v, w]   # Path length to be tested
                    if Duw < D[w]           # If new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]        # NP(u->w) = NP of new path
                        P[w, :] .= false
                        P[w, v] = true      # v is the only predecessor
                    elseif Duw == D[w]      # If new u->w equal to old
                        NP[w] += NP[v]      # NP(u->w) sum of old and new
                        P[w, v] = true      # v is also a predecessor
                    end
                end
            end

            # Calculate minD with an initial value for the reduce operation
            minD = sum(S) > 0 ? minimum(D[S]) : []
            if isempty(minD)
                break
            elseif isinf(minD)
                Q[1:q] .= findall(isinf.(D))
                break
            end
            V = findall(x -> x == minD, D)
        end

        DP = zeros(Int, n)   # Dependency
        for w in Q[1:n-1]
            BC[w] += DP[w]
            for v in findall(x -> x != 0, P[w, :])
                DP[v] += (1 + DP[w]) * NP[v] // NP[w]
            end
        end
    end

    return BC
end


# include("test_data.jl")
# W_Y, D, A_init = load_weight_test_data()



#weight_conversion(A)
# clustering_coef_wu(weight_conversion(W_Y))
# betweenness_wei(weight_conversion(W_Y))