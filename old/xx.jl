using DataFrames
using HDF5
using Statistics: mean

datasets = [
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx",
    "/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hpc",
    "/store/DAMTPEGLEN/mw894/data/Demas2006/",
    "/store/DAMTPEGLEN/mw894/data/Maccione2014",
]

df = DataFrame(
    dset_name=String[],
    sample_name=String[],
    div=Int[],
    mean_firing_rate=Float64[]
)

for p_dir in datasets
    # get 
    dset_name = replace(p_dir, "/store/DAMTPEGLEN/mw894/data/" => "")

    # get files
    recording_files = filter(file -> endswith(file, ".h5") && startswith(file, "sample_"), readdir(p_dir))

    # start with sample and h5 extension
    for sample_name in recording_files
        # read in the age
        file = h5open(joinpath(p_dir, sample_name), "r")
        div = read(file, "meta/age")[1]
        avg_firing_rate = mean(read(file, "/summary/frate"))
        close(file)

        # add row
        push!(df, (dset_name, sample_name, div, avg_firing_rate))
    end
end

# 1. Number of sampels at each div
sample_age = combine(groupby(df, [:dset_name, :div]), nrow => :Count)
sample_age = sort(sample_age, :div)