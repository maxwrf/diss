include("gnm.jl")

using HDF5
using .GNM_Mod
using StatsBase: sample
using LinearAlgebra: triu, Symmetric

function main(test_path::Union{String,Nothing}=nothing)
    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end

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
    org_file_name = read_attribute(meta_group, "org_file_name")
    close(file)

    println("Dataset: ", data_set_name)
    println("Group ID: ", group_id)
    println("Model: ", model_name)
    println("Runs: ", size(param_space, 1))


    # downsample
    m = sum(A_Y) / 2
    if m > 1024
        println("--DOWNSAMPLE--")

        # downsample
        edges = findall(==(1), triu(A_Y, 1))
        removal_indices = sample(edges, Int(m - 1024); replace=false)
        A_Y[removal_indices] .= 0
        A_Y = Symmetric(A_Y, :U)
        m = sum(A_Y) / 2

        # redo the init matrix
        A_init = zeros(size(A_Y))
        edges = findall(==(1), triu(A_Y, 1))
        init_edges = sample(edges, Int(round(m * 0.2)); replace=false)
        A_init[init_edges] .= 1
        A_init = Symmetric(A_init, :U)

        A_Y = Matrix(A_Y)
        A_init = Matrix(A_init)
    end

    println("M: ", m)
    println("M init: ", sum(A_init) / 2)

    # run model
    model = GNM_Mod.GNM(A_Y, D, A_init, param_space, model_id)
    @time GNM_Mod.generate_models(model)

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
    attributes(meta_group)["org_file_name"] = org_file_name

    close(file)
end


main("/store/DAMTPEGLEN/mw894/slurm/Charlesworth2015/sample_00001.dat")
#main("/store/DAMTPEGLEN/mw894/data/Maccione2014/sample_00416.dat")