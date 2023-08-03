

function main()
    n_runs = 3000
    limits_one = [-7.0, 7.0]
    limits_two = [-7.0, 7.0]
    limits_three = [0.02, 0.1]
    limits_four = [0.8, 1.0]

    η = range(limits_one[1], stop=limits_one[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    γ = range(limits_two[1], stop=limits_two[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    α = range(limits_three[1], stop=limits_three[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))
    ω = range(limits_four[1], stop=limits_four[2], length=max(floor(Int64, (n_runs)^(1 / 4)), 2))

    params_tups = unique(collect(Iterators.product(η, γ, α, ω)), dims=1)
    params_tups = vec(permutedims(params_tups, [4, 3, 2, 1]))

    params = zeros(length(params_tups), 4)
    for iParams in 1:length(params_tups)
        params[iParams, 1] = params_tups[iParams][1]
        params[iParams, 2] = params_tups[iParams][2]
        params[iParams, 3] = params_tups[iParams][3]
        params[iParams, 4] = params_tups[iParams][4]
    end
end