function load_test_data(n_samples::Int=-1)
    pData = "/Users/maxwuerfek/code/diss/data/example_binarised_connectomes.mat"
    pDist = "/Users/maxwuerfek/code/diss/data/dk_coordinates.mat"

    A = matread(pData)["example_binarised_connectomes"]
    D = matread(pDist)["coordinates"]
    D = pairwise(Euclidean(), D, dims=1)

    # TODO: Is this the correct #
    A_init = Float64.(mean(A, dims=1) .== 0.2)
    A_init = repeat(A_init, 270)

    if n_samples != -1
        A = A[1:n_samples, :, :]
        A_init = A_init[1:n_samples, :, :]
    end

    return A, D, A_init
end