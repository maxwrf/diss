include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

using .GNM_Mod

const PARAMS = Dict(
    "out_dir" => "/Users/maxwuerfek/code/diss/gnm/test/",
    "n_samples" => 2,
    "n_runs" => 1000
)

function main()
    As, D, A_inits = load_test_data(PARAMS["n_samples"])
    param_space = generate_param_space(PARAMS["n_runs"])

    K = zeros(length(MODELS), Int(size(As, 1)), Int(size(param_space, 1)), 4)

    for i_sample in axes(As, 1)
        A = As[i_sample, :, :]
        A_init = A_inits[i_sample, :, :]
        for (model_id, _) in MODELS
            println("Sample: ", i_sample, " Model: ", model_id)
            elapsed_time = @elapsed begin
                model = GNM_Mod.GNM(A, D, A_init, param_space, model_id)
                GNM_Mod.generate_models(model)
                K[model_id, i_sample, :, :] = model.K
            end
            println("Elapsed: $elapsed_time seconds")
        end
    end

    # save the results
    file = h5open(PARAMS["out_dir"] * "results.h5", "w")
    meta_group = create_group(file, "meta")
    attributes(meta_group)["data_set_id"] = "test"
    attributes(meta_group)["data_set_name"] = "test"
    attributes(meta_group)["group_id"] = "test"

    write(file, "param_space", param_space)

    result_group = create_group(file, "results")
    for (model_id, _) in MODELS
        # number of samples x number of parameters x number of eval functions
        K_model = K[model_id, :, :, :]
        write(result_group, string(model_id), K_model)
    end
    close(file)
end

main()