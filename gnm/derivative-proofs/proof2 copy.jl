using LinearAlgebra: Diagonal, I, norm, lu
using ExponentialUtilities
using ForwardDiff: gradient, derivative, jacobian
using Polynomials: fit

W = [0.0 0.8 0.0
    0.8 0.0 0.2
    0.0 0.2 0.0]

W = [1 0.2
    0.2 0]

function _diff_pade3(A, E, ident)
    b = (120.0, 60.0, 12.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    U = A * (b[4] * A2 + b[2] * ident)
    V = b[3] * A2 + b[1] * ident
    Lu = A * (b[4] * M2) + E * (b[4] * A2 + b[2] * ident)
    Lv = b[3] * M2
    return U, V, Lu, Lv
end

function _diff_pade5(A, E, ident)
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    U = A * (b[6] * A4 + b[4] * A2 + b[2] * ident)
    V = b[5] * A4 + b[3] * A2 + b[1] * ident
    Lu = A * (b[6] * M4 + b[4] * M2) + E * (b[6] * A4 + b[4] * A2 + b[2] * ident)
    Lv = b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

function _diff_pade7(A, E, ident)
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    U = A * (b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * ident)
    V = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    Lu = A * (b[8] * M6 + b[6] * M4 + b[4] * M2) + E * (b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * ident)
    Lv = b[7] * M6 + b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

function _diff_pade9(A, E, ident)
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    A8 = A4 * A4
    M8 = A4 * M4 + M4 * A4
    U = A * (b[10] * A8 + b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * ident)
    V = b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    Lu = A * (b[10] * M8 + b[8] * M6 + b[6] * M4 + b[4] * M2) + E * (b[10] * A8 + b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * ident)
    Lv = b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

function _diff_pade13(A, E, ident)
    # pade order 13
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    b = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0,
        16380.0, 182.0, 1.0)
    W1 = b[14] * A6 + b[12] * A4 + b[10] * A2
    W2 = b[8] * A6 + b[6] * A4 + b[4] * A2 + b[2] * ident
    Z1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    Z2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    W = A6 * W1 + W2
    U = A * W
    V = A6 * Z1 + Z2
    Lw1 = b[14] * M6 + b[12] * M4 + b[10] * M2
    Lw2 = b[8] * M6 + b[6] * M4 + b[4] * M2
    Lz1 = b[13] * M6 + b[11] * M4 + b[9] * M2
    Lz2 = b[7] * M6 + b[5] * M4 + b[3] * M2
    Lw = A6 * Lw1 + M6 * W1 + Lw2
    Lu = A * Lw + E * W
    Lv = A6 * Lz1 + M6 * Z1 + Lz2
    return U, V, Lu, Lv
end

ell_table_61 = (nothing, 2.11e-8, 3.56e-4, 1.08e-2, 6.49e-2, 2.00e-1, 4.37e-1,
    7.83e-1, 1.23e0, 1.78e0, 2.42e0, 3.13e0, 3.90e0, 4.74e0, 5.63e0,
    6.56e0, 7.52e0, 8.53e0, 9.56e0, 1.06e1, 1.17e1
);


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




node_strengths = dropdims(sum(W, dims=2), dims=2)
node_strengths[node_strengths.==0] .= 1e-5
S = sqrt(inv(Diagonal(node_strengths)))
R, L = frechet_algo(S * W * S, [1.0 0; 0 0])
#display(R)
display(L)





function func_gx(W)
    node_strengths = dropdims(sum(W, dims=2), dims=2)
    node_strengths[node_strengths.==0] .= 1e-5
    S = sqrt(inv(Diagonal(node_strengths)))
    return S * W * S
end

function func_fg(W)
    return exponential!(copyto!(similar(W), W), ExpMethodGeneric())
end

function forward_diff_j(f::Function, W::Matrix{Float64})
    return jacobian(f, W)
end


dgx = forward_diff_j(func_gx, W)
dfg = forward_diff_j(func_fg, S * W * S)
sum(dfg * dgx, dims=1)




