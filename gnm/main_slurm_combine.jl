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