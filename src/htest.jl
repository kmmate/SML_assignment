#!/usr/local/bin/julia

#=
Performs permutation test for independence
@author: Mate Kormos
=#

using Statistics, StatsBase, Distributed, Random, Printf
using Distances, Statistics

"""
    dcor(x::Array{<:Real, 1}, y::Array{<:Real, 1})

Compute distance correlation.

##### Returns
`dc`::Float64 : distance correlation between `x` and `y`

##### References
- Szekely et al. (2007) Measuring and testing dependence by
correlation of distances.
"""
function dcor(x::Array{<:Real, 1}, y::Array{<:Real, 1})
    (length(x) != length(y)) && (error("`x` and `y` must have same length"))
    n = length(x)  # number of obs
    # compute distance matrices
    a = pairwise(Euclidean(), reshape(x, n, 1), dims=1)
    b = pairwise(Euclidean(), reshape(y, n, 1), dims=1)
    # centering them
    a_colmeans = mean(a, dims=1)[:]; b_colmeans = mean(b, dims=1)[:]
    a_rowmeans = mean(a, dims=2)[:]; b_rowmeans = mean(b, dims=2)[:]
    a_mean = mean(a); b_mean = mean(b)  # grand mean
    for j in 1:n
        for k in 1:n
            a[j, k] = a[j, k] - a_rowmeans[j] - a_colmeans[k] + a_mean
            b[j, k] = b[j, k] - b_rowmeans[j] - b_colmeans[k] + b_mean
        end
    end
    vsq_xy = sum(a .* b) / (n ^ 2)
    vsq_x = sum(a .* a) / (n ^ 2)
    vsq_y = sum(b .* b) / (n ^ 2)
    dc = sqrt(vsq_xy / sqrt(vsq_x * vsq_y))
    return dc
end
"""
    perm_test(x::Array{<:Real, 1}, y::Array{<:Real, 1})

Permutation test for independence.
"""
function perm_test(x::Array{<:Real, 1}, y::Array{<:Real, 1})
    depmeasure = cor  # measure of dependence
    reps = 2000  # number of permutations
    test_stat = depmeasure(x, y)
    s = 0.0
    for rep in 1:reps
        # permute observations
        x_perm = shuffle(x)
        # compute test statistics
        dm = depmeasure(x_perm, y)
        # compare it to original estimate
        s += abs2(test_stat) <= abs2(dm)
    end
    pvalue = s/reps
    # write to latex
    file = "../latex/ptest_results.tex"
    s = @sprintf("test statistics of %.4f, with corresponding \$ p\$-value %.4f",
                 test_stat, pvalue)
    open(file, "w") do io
        write(io, s)
    end
    #return (test_stat, pvalue)
end

#n = 500
#x = randn(n)
#y = x .^ 1
#(test_stat, pvalue) = perm_test(x, y)
