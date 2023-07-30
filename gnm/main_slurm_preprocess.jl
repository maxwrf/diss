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

    @time generate_inputs(file_path)
end

#main("/store/DAMTPEGLEN/mw894/data/Demas2006/sample_00009.h5")
main()
