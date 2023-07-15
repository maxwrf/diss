include("gnm.jl")

using HDF5
using .GNM_Mod

function main()
    if length(ARGS) != 1
        error("Please provide a data file path.")
    end
    file_path = ARGS[1]

    # read data
    file = h5open(file_path, "r")
    A_Y = read(file, "A_Y")
    A_init = read(file, "A_init")
    D = read(file, "D")
    param_space = read(file, "param_space")

    # read meta data
    meta_group = file["meta"]
    d_set_id = read_attribute(meta_group, "data_set_id")
    data_set_name = read_attribute(meta_group, "data_set_name")
    group_id = read_attribute(meta_group, "group_id")
    model_id = read_attribute(meta_group, "model_id")
    model_name = read_attribute(meta_group, "model_name")
    close(file)

    # run model
    model = GNM_Mod.GNM(A_Y, D, A_init, param_space, model_id)
    GNM_Mod.generate_models(model)

    # save results
    res_file_path = replace(file_path, r"\.dat$" => ".res")
    file = h5open(res_file_path, "w")

    write(file, "K", model.K)
    write(file, "param_space", param_space)
    meta_group = create_group(file, "meta")

    attributes(meta_group)["data_set_id"] = d_set_id
    attributes(meta_group)["data_set_name"] = data_set_name
    attributes(meta_group)["group_id"] = group_id
    attributes(meta_group)["model_id"] = model_id
    attributes(meta_group)["model_name"] = model_name

    close(file)
end

main()
