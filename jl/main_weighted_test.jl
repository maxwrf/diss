include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

using .GNM_Mod

function main()
    models = ["matching"]

    # load synthetic data
    W_Y, D, A_init = load_weight_test_data()
    A_Y = Float64.(W_Y .> 0)

    # create param space
    eta = -3.2
    gamma = 0.38
    alpha = 0.05
    omega = 0.9
    params = [eta, gamma, alpha, omega]
    params = reshape(params, 1, 4)

    # weighted model parameters
    start_edge = 0
    opti_func = 0
    opti_samples = 5
    opti_resolution = 0.05

    # TODO: Add back number of samples
    K = zeros(1, length(models), Int(size(params, 1)), 4)

    for i_sample in 1:1
        for i_model in 1:length(models)
            # To be removed
            i_model = 3
            println("Sample: ", i_sample, " Model: ", i_model)

            model = GNM_Mod.GNM(A_Y, D, A_init, params, i_model, true, W_Y,
                start_edge, opti_func, opti_samples, opti_resolution)

            # generate_models(model)
            # K[i_sample, i_model, :, :] = model.K
        end
    end

end

main()