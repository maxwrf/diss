include("gnm_analyze.jl")

function main(result_dir::String)
    group_res_ps = filter(name -> endswith(name, ".h5"), readdir(result_dir))
    group_res_ps = map(name -> joinpath(result_dir, name), group_res_ps)

    for group_res_p in group_res_ps
        analyze(res_files)
    end
end


main()