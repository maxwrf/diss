include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

using .GNM_Mod

function main()
    models = ["spatial", "neighbors", "matching", "clu-avg", "clu-min",
        "clu-max", "clu-dist", "clu-prod", "deg-avg", "deg-min", "deg-max",
        "deg-dist", "deg-prod"
    ]

    As, D, A_inits = load_test_data(2)
    params = generate_param_space(1000)

    K = zeros(Int(size(As, 1)), length(models), Int(size(params, 1)), 4)

    for i_sample in axes(As, 1)
        A = As[i_sample, :, :]
        A_init = A_inits[i_sample, :, :]
        for i_model in 1:length(models)
            println("Sample: ", i_sample, " Model: ", i_model)
            model = GNM_Mod.GNM(A, D, A_init, params, i_model)
            GNM_Mod.generate_models(model)
            K[i_sample, i_model, :, :] = model.K
        end
    end

    out_dir = "/Users/maxwuerfek/code/diss/jl/test/"
    save_K(out_dir, K, params)
end

main()