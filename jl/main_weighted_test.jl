include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

function main()
    models = ["matching"]

    # load synthetic data
    W_Y, D, A_init = load_weight_test_data()
    A_Y = Float64.(A_Y .> 0)

    # create param space
    eta = -3.2
    gamma = 0.38
    alpha = 0.05
    omega = 0.9
    params = [[eta, gamma, alpha, omega]]

    # weighted model parameters
    start = 0
    opti_func = 0
    opti_samples = 5
    opti_resolution = 0.05

    K = zeros(Int(size(As, 1)), length(models), Int(size(params, 1)), 4)


    for i_sample in 1:1
        for i_model in 1:length(models)
            # To be removed
            i_model = 3

            println("Sample: ", i_sample, " Model: ", i_model, "\n")
            model = Weighted_GNM(A, D, A_init, params, i_model)
            generate_models(model)
            K[i_sample, i_model, :, :] = model.K
        end
    end

end

main()