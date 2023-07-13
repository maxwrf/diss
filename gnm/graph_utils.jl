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