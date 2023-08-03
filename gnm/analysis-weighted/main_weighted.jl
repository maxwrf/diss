using Random
using JLD2

include("../test_data.jl")
include("../gnm_utils.jl")
include("../gnm.jl")


function save_object(filename)
    n_runs = 3000
    i_model = 1

    # load synthetic data
    W_Y, D, A_init = load_weight_test_data()
    A_Y = Float64.(W_Y .> 0);

    Random.seed!(1234)



    model = GNM_Mod.GNM(A_Y, D, A_init, params, 1, true, W_Y, 0)
    GNM_Mod.generate_models(model)
    save_object("model_"* string(i_model)* "".jld2", model)
end
