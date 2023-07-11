include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

function main()
    models = ["spatial"]
    As, D, A_inits = load_test_data(2)
    params = generate_param_space(1000)

    K = zeros(Int(size(As, 1)), length(models), Int(size(params, 1)), 4)

    for iSample in 1:size(As, 1)
        A = As[iSample, :, :]
        A_init = A_inits[iSample, :, :]
        for iModel in [13]
            println("Sample: ", iSample, " Model: ", iModel, "\n")
            K[iSample, 1, :, :] = generate_models(A, D, A_init, params, iModel)
        end
    end

    out_dir = "/Users/maxwuerfek/code/diss/jl/test/"
    save_K(out_dir, K, params)
end

main()