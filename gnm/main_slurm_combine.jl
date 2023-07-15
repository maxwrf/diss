include("slurm.jl")

const PARAMS = Dict(
    "cluster" => false,
    "d_set" => 1
)

const DATA_SETS = Dict(
    1 => "Charlesworth2015",
    2 => "Hennig2011",
    3 => "Demas2006",
    4 => "Maccione2014"
)

function main()
    # set paths 
    if PARAMS["cluster"]
        in_dir = "/store/DAMTPEGLEN/mw894/slurm/" * DATA_SETS[PARAMS["d_set"]]
    else
        in_dir = "/Users/maxwuerfek/code/diss/gnm/slurm/" * DATA_SETS[PARAMS["d_set"]]
    end

    combine_res_files(in_dir)
end

main();