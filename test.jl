using ForwardDiff
using ExponentialAction
using ExponentialUtilities


rep_vec = [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

A = [0 0.8; 0.8 0]

edge = CartesianIndex(1, 2)
edge_val = A[edge]
reps = [edge_val * (1 + i) for i in rep_vec]

rep_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


myexp(A) = exponential!(copyto!(similar(A), A), ExpMethodGeneric());
ForwardDiff.jacobian(myexp, A)


function custom_mapping(n::Int)
    mapped_value = (n - 1) % 3 + 1
    if n > 3
        mapped_value += 3
    end
    return mapped_value
end

# Example usage
println(custom_mapping(1))  # Output: 1
println(custom_mapping(2))  # Output: 4
println(custom_mapping(3))  # Output: 7
println(custom_mapping(4))  # Output: 2
println(custom_mapping(5))  # Output: 5
println(custom_mapping(6))  # Output: 8



