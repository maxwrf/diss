include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

using .GNM_Mod
using Random

function main()
    models = ["matching"]
    Random.seed!(1234)

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
    start_edge = 0

    # TODO: Add back number of samples
    K = zeros(1, length(models), Int(size(params, 1)), 4)
    K_W = zeros(1, length(models), Int(size(params, 1)), 3)

    for i_sample in 1:1
        for i_model in 1:length(models)
            # To be removed
            i_model = 3
            println("Sample: ", i_sample, " Model: ", i_model)
            elapsed_time = @elapsed begin
                model = GNM_Mod.GNM(
                    A_Y,
                    D,
                    A_init,
                    params,
                    i_model,
                    true,
                    W_Y,
                    start_edge
                )
                #model.m = 80
                GNM_Mod.generate_models(model)
                # TODO: Need to change for prod
                K[i_sample, 1, :, :] = model.K
                K_W[i_sample, 1, :, :] = model.K_W
            end
            println("Elapsed: $elapsed_time seconds")
        end
    end
    println(K)
    println(K_W)
end

main()