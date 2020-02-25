#!/usr/local/bin/julia

#=

Main code for Supervised Machine Learning Final Assignment. Due: 01 March 2020.
@author: Mate Kormos

=#

#using DataFrames, Statistics

include("data_preprocessing.jl")
include("htest.jl")
include("estimators.jl")
include("makereport.jl")


"""
	main()

Main function
"""
function main()
	Random.seed!(09865757545657)
	# Get data
	df = read_data()
	summary_stats(df)
	y = df.bweight  # outcome
	d = df.smoke  # treatment
	x = Array(df[:,Not([:bweight, :smoke])])  # covariates
	x_poly = polynomial_features(x, 3)[:,2:end]  # add polynomials, drop intercept
	N = length(y)
	# Permutation test
	perm_test(y, d)
	# ATE estimation
	K = 5  # number of folds in crossfitting
	(theta_hats, sigmasq_hats) = ate_estimator(y, d, x_poly, K)
	makereport(theta_hats, sigmasq_hats, N)
end
main()
