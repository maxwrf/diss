using HDF5

include("gnm.jl")
include("gnm_utils.jl")

function main(test_path::Union{String,Nothing}=nothing)
    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end

    # only if we have not done this one
    res_file_path = replace(file_path, r"\.h5$" => ".res")
    if isfile(res_file_path)
        println("File already exists: ", res_file_path)
        return
    end

    # read data
    file = h5open(file_path, "r")
    A_Y = read(file, "A_Y")
    W_Y = read(file, "W_Y")
    A_init = read(file, "A_init")
    D = read(file, "D")
    params = read(file, "params")
    println("Params: ", params)


    model = GNM_Mod.GNM(A_Y, D, A_init, reshape(params, (1, 4)), 4, true, W_Y, 0)
    GNM_Mod.generate_models(model)

    # store results
    file = h5open(res_file_path, "w")

    write(file, "K_pcomb", model.K[1, :])
    write(file, "K_W_pcomb", model.K_W[1, :, :])
    write(file, "A_final_pcomb", model.A_final[1, :, :])
    write(file, "W_final_pcomb", model.W_final[1, :, :])

    meta_group = create_group(file, "meta")
    attributes(meta_group)["params"] = params
    attributes(meta_group)["file_name"] = res_file_path

    close(file)
end

main("/store/DAMTPEGLEN/mw894/data/weighted/sample_00873.h5")