using Distributions
using StatsBase

# Generate two random samples
sample1 = rand(Normal(0, 1), 100)
sample2 = rand(Normal(0, 1), 100)

# Perform the two-sample KS test
ks_statistic, p_value = ks_test(sample1, sample2)

println("KS statistic: ", ks_statistic)
println("P-value: ", p_value)
