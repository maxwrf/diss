include("gnm.jl")
include("test_data.jl")
include("gnm_utils.jl")

using .GNM_Mod
using Random

function main(file_path::String)
    # read data
    file = h5open(file_path, "r")
    A_Y = read(file, "A_Y")
    A_init = read(file, "A_init")
    D = read(file, "D")
    sttc = read(file, "sttc")

    # read meta data
    meta_group = file["meta"]
    d_set_id = read_attribute(meta_group, "data_set_id")
    data_set_name = read_attribute(meta_group, "data_set_name")
    group_id = read_attribute(meta_group, "group_id")
    model_id = read_attribute(meta_group, "model_id")
    model_name = read_attribute(meta_group, "model_name")
    org_file_name = read_attribute(meta_group, "org_file_name")
    close(file)

    println("Dataset: ", data_set_name)
    println("Group ID: ", group_id)
    println("Model: ", model_name)
    println("Runs: ", size(param_space, 1))


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


    model = GNM_Mod.GNM(A_Y, D, A_init, params, model_id, true, W_Y, start_edge)
    model.m = 80
    GNM_Mod.generate_models(model)
    # TODO: Need to change for prod
    K[i_sample, 1, :, :] = model.K
    K_W[i_sample, 1, :, :] = model.K_W



    println(K)
    println(K_W)
end

main("/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx/sample_00828.dat")