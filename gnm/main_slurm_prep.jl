include("slurm.jl")

const PARAMS = Dict(
    "cluster" => false,
    "d_set" => 3,
    "corr_cutoff" => 0.2,
    "n_samples" => -1,
    "n_runs" => 10000,
    "dt" => 0.05
)

const DATA_SETS = Dict(
    1 => "Charlesworth2015",
    2 => "Hennig2011",
    3 => "Demas2006",
    4 => "Maccione2014"
)

const MEA_TYPES = Dict(
    1 => "MCS_8x8_200um",
    2 => "MCS_8x8_100um",
    3 => "APS_64x64_42um"
)

function main()
    # get mea type
    if PARAMS["d_set"] == 1
        mea_type = 1
    elseif PARAMS["d_set"] == 3
        mea_type = 2
    elseif PARAMS["d_set"] == 4
        mea_type = 3
    end

    # set paths 
    if PARAMS["cluster"]
        in_dir = "/store/DAMTPEGLEN/mw894/data/" * DATA_SETS[PARAMS["d_set"]]
        out_dir = "/store/DAMTPEGLEN/mw894/slurm/" * DATA_SETS[PARAMS["d_set"]]
    else
        in_dir = "/Users/maxwuerfek/code/diss/data/" * DATA_SETS[PARAMS["d_set"]]
        out_dir = "/Users/maxwuerfek/code/diss/gnm/slurm/" * DATA_SETS[PARAMS["d_set"]]
    end

    println("Dataset: ", DATA_SETS[PARAMS["d_set"]])
    println("MEA type: ", MEA_TYPES[mea_type])
    println("Runs: ", PARAMS["n_runs"])
    println("Samples: ", PARAMS["n_samples"])

    @time generate_inputs(
        in_dir,
        out_dir,
        PARAMS["n_samples"],
        PARAMS["n_runs"],
        PARAMS["d_set"],
        mea_type,
        PARAMS["dt"],
        PARAMS["corr_cutoff"]
    )
end

main()
