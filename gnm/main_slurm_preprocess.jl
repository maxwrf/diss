using JSON

include("slurm.jl")

const config = JSON.parsefile("/home/mw894/diss/gnm/config.json")

function main(test_path::Union{String,Nothing}=nothing)
    println("Check the config!")

    if length(ARGS) == 1
        file_path = ARGS[1]
    elseif test_path !== nothing
        file_path = test_path
    else
        error("Please provide a data file path.")
    end


    println("File: ", file_path)
    println("Runs: ", config["params"]["n_runs"])

    @time generate_inputs(
        file_path,
        config["params"]["n_runs"],
        config["params"]["d_set"],
        config["params"]["dt"]
    )
end

#main("/Users/maxwuerfek/code/diss/data/Maccione2014/sample_00045.h5")
main()
