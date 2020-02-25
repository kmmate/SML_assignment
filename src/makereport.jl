#!/usr/local/bin/julia

#=
Make report from estimated ATEs and Var[ATE]s in each partition.
@author: Mate Kormos
=#
using Printf

function makereport(theta_hats::Array{Float64,1},
                    sigmasq_hats::Array{Float64, 1},
                    N::Int64)
    (length(theta_hats) != length(sigmasq_hats)) &&
        throw(DimensionMismatch(
        "`theta_hats` and `sigmasq_hats` must have same length"))
    K = length(theta_hats)
    # theta_hats: write to latex
    file = "../latex/theta_hats.tex"
    s = join(map(x->@sprintf("%.2f", x), theta_hats), ", ")
    open(file, "w") do io
        write(io, s)
    end
    # theta_hat: write to latex
    theta_hat = sum(theta_hats) / K
    file = "../latex/theta_hat.tex"
    s = @sprintf("%.4f", theta_hat)
    open(file, "w") do io
        write(io, s)
    end
    # sigmasq_hats: write to latex
    file = "../latex/sigmasq_hats.tex"
    s = join(map(x->@sprintf("%.2e", x), sigmasq_hats), ", ")
    open(file, "w") do io
        write(io, s)
    end
    # sigmasq_hat: write to latex
    sigmasq_hat = sum(sigmasq_hats) / K
    file = "../latex/sigmasq_hat.tex"
    s = @sprintf("%.4e", sigmasq_hat)
    open(file, "w") do io
        write(io, s)
    end
    ci_low = theta_hat - 1.96 * sqrt(sigmasq_hat / N)
    ci_high = theta_hat + 1.96 * sqrt(sigmasq_hat / N)
    file = "../latex/confint.tex"
    s = @sprintf("\$[%.4f, %.4f]\$", ci_low, ci_high)
    open(file, "w") do io
        write(io, s)
    end
end

#Z = 200.0 .+ randn(5)
#W = randn(5) .^ 2
#makereport(Z, W, 5)
