include("gnm.jl")
include("gnm_utils.jl")

using HDF5
using .GNM_Mod
using Random
using JSON


const config = JSON.parsefile("/home/mw894/diss/gnm/config.json")

function main(file_path::String)
    # get the best eta gamma combination from the standard GNM run
    res_path = replace(file_path, ".dat" => ".res")
    println(res_path)
    res_file = h5open(res_path, "r")
    K = read(res_file, "K")
    param_space = read(res_file, "param_space")
    close(res_file)

    K_MAX = dropdims(maximum(K, dims=2); dims=2)
    _, top_idx = findmin(K_MAX)
    etas = repeat([param_space[top_idx, 1]], config["params_w"]["n_runs"])
    gammas = repeat([param_space[top_idx, 2]], config["params_w"]["n_runs"])

    # a little clean up to be sure
    K = nothing
    param_space = nothing
    K_MAX = nothing

    # compute the omega and alpha space
    alpha_omga_space = generate_param_space(
        config["params_w"]["n_runs"],
        [config["params_w"]["alpha_min"], config["params_w"]["alpha_max"]],
        [config["params_w"]["omega_min"], config["params_w"]["omega_max"]]
    )

    param_space = hcat(etas, gammas, alpha_omga_space[:, 1], alpha_omga_space[:, 2])

    # read data from the .dat file
    file = h5open(file_path, "r")
    A_Y = read(file, "A_Y")
    A_init = read(file, "A_init")
    D = read(file, "D")
    sttc = read(file, "sttc")
    W_Y = abs.(A_Y .* sttc) # TODO: check if this is correct

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
    println("DIV: ", group_id)
    println("Model: ", model_name)
    println("Runs: ", size(param_space, 1))

    # generate model
    model = GNM_Mod.GNM(A_Y, D, A_init, param_space, model_id, true, W_Y, 0)
    GNM_Mod.generate_models(model)

    # save results
    res_file_path = replace(file_path, r"\.dat$" => ".wres")
    file = h5open(res_file_path, "w")

    write(file, "K", model.K)
    write(file, "K_W", model.K_W)
    write(file, "param_space", param_space)

    meta_group = create_group(file, "meta")
    attributes(meta_group)["data_set_id"] = d_set_id
    attributes(meta_group)["data_set_name"] = data_set_name
    attributes(meta_group)["group_id"] = group_id
    attributes(meta_group)["model_id"] = model_id
    attributes(meta_group)["model_name"] = model_name
    attributes(meta_group)["org_file_name"] = org_file_name
    close(file)
end

main("/store/DAMTPEGLEN/mw894/data/Demas2006/sample_00242.dat")