using JSON

include("slurm.jl")

const config = JSON.parsefile("gnm/config.json")

function main(test_path::Union{String,Nothing}=nothing)
    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end

    # get mea type
    if ((config["params"]["d_set"] == "1") || (config["params"]["d_set"] == "2"))
        mea_type = "1"
    elseif config["params"]["d_set"] == "3"
        mea_type = "2"
    else
        error("MEA type not defined for data set.")
    end


    println("File: ", file_path)
    println("MEA type: ", config["mea_types"][mea_type])
    println("Runs: ", config["params"]["n_runs"])

    @time generate_inputs(
        file_path,
        config["params"]["n_runs"],
        config["params"]["d_set"],
        mea_type,
        config["params"]["dt"]
    )
end

#main("/Users/maxwuerfek/code/diss/data/Maccione2014/sample_00045.h5")
main()
