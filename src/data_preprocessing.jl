#!/usr/local/bin/julia

# Read data, create summary table
using CSV, DataFrames, Latexify, LaTeXStrings, Statistics, IterTools
"""
	read_data()

Read data set
"""
function read_data()
	data_path = "../data/"
	df_y = CSV.read(data_path * "y.csv", delim=',')
	df_d = CSV.read(data_path * "d.csv", delim=',')
	df_x = CSV.read(data_path * "x.csv", delim=',')
	df = hcat(df_y, df_d, df_x)
	return(df)
end


"""
	summary_stats(df::DataFrame)

Print summary statistics to latex file.
"""
function summary_stats(df::DataFrame)
	# summary statistics grouped by smoke
	#summary_table = by(df, :smoke) do dff
	#	DataFrame(count=length(dff.bweight),
	#			  min=minimum(dff.bweight), max=maximum(dff.bweight),
	#			  med=median(dff.bweight), mean=mean(dff.bweight),
	#			  std=std(dff.bweight))
	#end
	stat_fns = [:length=>length, :min, :median, :max, :mean, :std]
	summary_control = describe(df[df.smoke .== 0, Not(:smoke)], stat_fns...)
	summary_treated = describe(df[df.smoke .== 1, Not(:smoke)], stat_fns...)
	# concat in zig-zag (control-treated) pattern
	summary_table = DataFrame()
	for i in 1:size(summary_treated)[1]
		summary_table = vcat(summary_table, summary_control[[i],:],
							 summary_treated[[i],:])
	end
	# rename variables
	for i in 1:size(summary_table)[1]
		if isodd(i)
			summary_table[i, :variable] = Symbol(
				string(summary_table[i, :variable]) * " nonsmokers")
		else
			summary_table[i, :variable] = Symbol(
				string(summary_table[i, :variable]) * " smokers")
		end
	end
	# write to latex
	col_names = ["Variable", "Count", "Minimum", "Median", "Maximum", "Mean",
				"Standard Deviation"]
	rename!(summary_table, map(Symbol, col_names))  # set column names
	latex_path = "../latex/"
	_latex_table(latex_path * "summary_table.tex", summary_table)
end


"""
	_latex_table(file::AbstractString, df::DataFrame; decimals=1,
					 adjustment=:l)

Write DataFrame to latex table.
"""
function _latex_table(file::AbstractString, df::DataFrame; decimals=1,
					 adjustment=:l)
	# latexify format, number of decimals to print
    set_default(fmt = "%.$(decimals)f")
	# latex command of tabular as string
    s = latexify(df, env=:tabular, latex=false, adjustment=adjustment)
    open(file, "w") do io
        write(io, s)
    end
end

"""
    polynomial_features(x::Array{<:Real}, poldegree::Int64)

Creates polynomial features from the data in `x`, including intercept and interactions

Based on [ScikitLearn.Preprocessing](https://github.com/cstjean/ScikitLearn.jl/blob/master/src/preprocessing.jl).

##### Arguments
-`x`:Array{<:Real} : Data array. Either ``n``-long Array{<:Real, 1} or ``n``-by-``d``
	Array{<:Real, 2} where ``d`` is the number of variables,
	``n`` is the number of observations
- `poldegree`::Int64 : Highest degree of the polynomial to be created.
	For eg. `poldegree`=3, `d=2` will include second order interactions, squares, cubic terms,
	the main effects and an intercept.

##### Returns
- `x_out`::Array{<:Real, 2} : Data array with the added polynomial feaures
"""
function polynomial_features(x::Array{<:Real}, poldegree::Int64)
	# sizes, checks
	n = size(x)[1]
	dim = ndims(x)
	if dim == 1
		d = 1
	elseif dim == 2
		d = size(x)[2]
	else
		error("`x` must be 1 or 2 dimensional")
	end
	if poldegree < 1
		error("`poldegree` cannot be smaller than 1")
	end

	# add features degree by degree
	no_outputvariables = sum([binomial(d, k) for k in 1:minimum([poldegree, d])]) +
		d * (poldegree - 1) + 1
	x_out = zeros(n, no_outputvariables)
	x_out[:, 1] = ones(n)  # intercept
	x_out[:, 2:d+1] = x  # main effects x ^ 0
	if poldegree == 1
		return x_out
	end
	idx = d + 2
	for degree in 2:poldegree
		# add main effects, ie x^2, x^3 etc., variable by variable
		for k = 1:d
			x_out[:, idx] = x[:, k] .^ degree
			idx += 1
		end
		# add interaction effects, ie x1*x2 etc.
		if degree <= d
			ss = IterTools.subsets(collect(1:d), degree)
			for s in ss
				x_out[:, idx] = prod(x[:, s], dims=2)[:]  # multipy columns elementwise
				idx +=1
			end
		end
	end
	return x_out
end

#df = read_data()
#summary_stats(df)
