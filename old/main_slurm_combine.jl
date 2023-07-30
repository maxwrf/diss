using JSON

include("slurm.jl")

const config = JSON.parsefile("/home/mw894/diss/gnm/config.json")

function main()
    # set paths 
    in_dir = "/store/DAMTPEGLEN/mw894/data/" * config["data_sets"][string(config["params"]["d_set"])]
    println("reading from: ", in_dir)
    combine_res_files(in_dir)
end

main();


function combine_res_files(in_dir::String)
    # get the names of all result files in the folder
    res_files = filter(name -> endswith(name, ".res"), readdir(in_dir))
    res_files = map(name -> joinpath(in_dir, name), res_files)

    # these variables will depend on the file
    K_all = Vector{Matrix{Float64}}() #  each K is param combs x 4 evals
    divs = Vector{String}()
    sample_names = Vector{String}()
    weeks = Vector{Int}()
    model_ids = Vector{Int}()

    # these variables are the same for all result files
    param_space = nothing
    dset_id = nothing
    data_set_name = nothing

    # For every result file in the folder
    for (i_res_files, res_file) in enumerate(res_files)
        file = h5open(res_file, "r")

        # the following variables are the same acros results and need to be read only once
        if i_res_files == 1
            param_space = read(file, "param_space")
            meta_group = file["meta"]
            dset_id = read_attribute(meta_group, "data_set_id")
            data_set_name = read_attribute(meta_group, "data_set_name")
        end

        # read meta data
        K = read(file, "K")
        push!(K_all, K)
        meta_group = file["meta"]
        div = read_attribute(meta_group, "group_id")
        sample_name = read_attribute(meta_group, "org_file_name")
        model_id = read_attribute(meta_group, "model_id")
        week = min(4, ceil.(Int, parse(Int, div) / 7))
        push!(divs, div)
        push!(sample_names, sample_name)
        push!(model_ids, model_id)
        push!(weeks, week)

        close(file)
    end

    println("Read ", length(res_files), " result files.")

    # write the combined results
    unique_weeks = unique(weeks)
    for week in unique_weeks
        # one file for every week with meta data in common & params in common
        file = h5open(joinpath(in_dir, "group_week_" * string(week) * ".h5"), "w")
        meta_group = create_group(file, "meta")
        attributes(meta_group)["week"] = week
        attributes(meta_group)["data_set_id"] = dset_id
        attributes(meta_group)["data_set_name"] = data_set_name
        write(file, "param_space", param_space)


        # will also store a mapper group for the actual names of the samples
        sample_map_group = create_group(file, "sample_map")
        div_map_group = create_group(file, "div_map")

        # one dataset for every model in the results group
        result_group = create_group(file, "results")
        for (model_id, _) in MODELS
            indices = findall((weeks .== week) .& (model_ids .== model_id))

            # if there was no model successfully build for the week
            if length(indices) == 0
                println("Model ", model_id, " not found in week ", week, ".")
                continue
            end

            # number of samples x number of parameters x number of eval functions
            K_group_model = Array{Float64}(undef, length(indices), size(param_space, 1), 4)
            for (i, idx) in enumerate(indices)
                K_group_model[i, :, :] = K_all[idx]
            end
            write(result_group, string(model_id), K_group_model)

            # write the sample names for whom this model was produced
            write(sample_map_group, string(model_id), sample_names[indices])
            write(div_map_group, string(model_id), divs[indices])
        end
        close(file)
    end

    println("Wrote ", length(unique_weeks), " week result files.")
end