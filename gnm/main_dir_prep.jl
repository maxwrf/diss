using Printf
using HDF5
using JSON

const config = JSON.parsefile("/home/mw894/diss/gnm/config.json")

function main(dataset_dir::String)
    recording_files = filter(file -> endswith(file, ".h5") && !startswith(file, "sample"), readdir(dataset_dir))
    for (i_recording, recording_file) in enumerate(recording_files)
        out_name = "sample_" * @sprintf("%05d", i_recording) * ".h5"
        out_path = joinpath(dataset_dir, out_name)
        Base.cp(joinpath(dataset_dir, recording_file), out_path, force=true)

        println(recording_file, " => ", out_path)

        # data set name
        prefix = "/store/DAMTPEGLEN/mw894/data/"
        data_set_name = replace(dataset_dir, prefix => "")
        reverse_data_sets = Dict(value => key for (key, value) in config["data_sets"])
        data_set_id = parse(Int, reverse_data_sets[data_set_name])

        # store the file name
        file = h5open(out_path, "cw")
        write(file, "meta/org_file_name", recording_file)
        write(file, "meta/data_set_name", data_set_name)
        write(file, "meta/data_set_id", data_set_id)
        close(file)
    end
end

# main("/store/DAMTPEGLEN/mw894/data/Maccione2014")
main("/store/DAMTPEGLEN/mw894/data/Demas2006")
main("/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx")
main("/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hpc")