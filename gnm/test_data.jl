using MAT
using Distances
using Statistics

function load_test_data(n_samples::Int=-1)
    pData = "/Users/maxwuerfek/code/diss/gnm/test/example_binarised_connectomes.mat"
    pDist = "/Users/maxwuerfek/code/diss/gnm/test/dk_coordinates.mat"

    A = matread(pData)["example_binarised_connectomes"]
    D = matread(pDist)["coordinates"]
    D = pairwise(Euclidean(), D, dims=1)

    A_init = Float64.(mean(A, dims=1) .== 0.2)
    A_init = repeat(A_init, 270)

    if n_samples != -1
        A = A[1:n_samples, :, :]
        A_init = A_init[1:n_samples, :, :]
    end

    return A, D, A_init
end

function load_weight_test_data()
    pData = "/Users/maxwuerfek/code/diss/gnm/test-weights/demo_data.mat"
    pData = "/home/mw894/diss/gnm/test-weights/demo_data.mat"
    data = matread(pData)
    D = data["demo_data"]["D"]
    A_init = data["demo_data"]["seed"]
    W_Y = data["demo_data"]["Wtgt"]
    return W_Y, D, A_init
end


