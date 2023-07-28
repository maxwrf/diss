using Printf
using HDF5

function main(dataset_dir::String)
    recording_files = filter(file -> endswith(file, ".h5") && !startswith(file, "sample"), readdir(dataset_dir))
    for (i_recording, recording_file) in enumerate(recording_files)
        out_name = "sample_" * @sprintf("%05d", i_recording) * ".h5"
        out_path = joinpath(dataset_dir, out_name)
        Base.cp(joinpath(dataset_dir, recording_file), out_path, force=true)

        println(recording_file, " => ", out_path)

        # store the file name
        file = h5open(out_path, "cw")
        write(file, "meta/org_file_name", recording_file)
        close(file)
    end
end

# main("/Users/maxwuerfek/code/diss/data/Charlesworth2015/ctx")
# main("/Users/maxwuerfek/code/diss/data/Charlesworth2015/hpc")
# main("/Users/maxwuerfek/code/diss/data/Demas2006/")

main("/store/DAMTPEGLEN/mw894/data/Maccione2014")
main("/store/DAMTPEGLEN/mw894/data/Demas2006")
main("/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx")
main("/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hpc")