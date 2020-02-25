#!/usr/local/bin/julia

#=
ATE and auxiliary estimators
@author: Mate Kormos
=#
using Random, DecisionTree, Lasso, StatsBase, NLsolve

"""
    ate_estimator(y::Array{T, 1} where T<:Real, d::Array{T, 1} where T<:Real,
        x::Array{T, 2} where T<:Real, K::Int64)

Estimate Average Treatment Effect (ATE) and the estimator's variance.

##### Returns
-`theta_hast`::Array{Float64,1}: theta_k estimates for k=1,2,...,K
-`sigmasq_hats`::Array{Float64,1}: sigmasq_k estimates for k=1,2,...,K
"""
function ate_estimator(y::Array{T, 1} where T<:Real,
                       d::Array{T, 1} where T<:Real,
                       x::Array{T, 2} where T<:Real,
                       K::Int64)
    N = length(y)
    part = _partition(N, K)  # partition label for each observation
    # estimates in each partition
    eta_hats, theta_hats = _ate_pointestimator(y,d,x,part, K)
    theta_hat = sum(theta_hats) / K
    sigmasq_hats = _ate_varestimator(eta_hats, theta_hat, y, d, x, part, K)
    return (theta_hats, sigmasq_hats)
end

"""
    _ate_varestimator(eta_hats::Array{Any,1}, theta_hat::Float64,
                                y::Array{T, 1} where T<:Real,
                                d::Array{T, 1} where T<:Real,
                                x::Array{T, 2} where T<:Real,
                                part::Array{Int64,1},
                                K::Int64)

Estimate the variance of estimated ATE.

##### Returns
-`sigmasq_hats`::Array{Float64,1}: sigmasq_k estimates for k=1,2,...,K
"""
function _ate_varestimator(eta_hats::Array{Any,1}, theta_hat::Float64,
                            y::Array{T, 1} where T<:Real,
                            d::Array{T, 1} where T<:Real,
                            x::Array{T, 2} where T<:Real,
                            part::Array{Int64,1},
                            K::Int64)
    println("Estimating sigmasq")
    sigmasq_hats = zeros(K)
    for k in 1:K
        sigmasq_hat = sum(_neyman_score(eta_hats[k], theta_hat, y[part .== k],
                                        d[part .== k], x[part .== k,:]) .^ 2)/K
        sigmasq_hats[k] = sigmasq_hat
    end
    return sigmasq_hats
end

"""
_ate_pointestimator(y::Array{T, 1} where T<:Real,
                             d::Array{T, 1} where T<:Real,
                             x::Array{T, 2} where T<:Real,
                             part::Array{Int64,1},
                             K::Int64)

Estimate Average Treatment Effect (ATE).

##### Returns
`eta_hats`::Array : (m_k, g_k) estimates for k=1,2,...,K
`theta_hats`::Array{Float64,4} : theta_k estimate for k=1,2,...,K.
"""
function _ate_pointestimator(y::Array{T, 1} where T<:Real,
                             d::Array{T, 1} where T<:Real,
                             x::Array{T, 2} where T<:Real,
                             part::Array{Int64,1},
                             K::Int64)
    println("Estimating theta:")
    theta_hats = zeros(K)  # estimated ate in each partition
    eta_hats = Array{Any,1}(undef, K)  # estimated eta in each partition
    for k in 1:K
        println("working on fold k = $(k)")
        # cross fitting: train ML on out-of-partition data
        println("...cross-fitting")
        eta_hat = _eta_estimator(y[part .!= k], d[part .!= k], x[part .!= k,:])
        # estimation via Neyman orthogonal scores
        println("...orthogonality scores")
        theta_hat = _theta_estimator(eta_hat, y[part .== k], d[part .== k],
                                     x[part .== k,:])
        eta_hats[k] = eta_hat
        theta_hats[k] = theta_hat
    end
    return (eta_hats, theta_hats)  # estimates for each partition
end

"""
    _eta_estimator(y::Array{T, 1} where T<:Real,
                        d::Array{T, 1} where T<:Real,
                        x::Array{T}  where T<:Real)

Estimate nuisance parameters with ML method.

##### Returns
`m_hat`: estimated m function x -> estimated P(D=1|x)
`g_hat`: estimated g function (D, x) -> estimated E[Y|D, x]
"""
function _eta_estimator(y::Array{T, 1} where T<:Real,
                        d::Array{T, 1} where T<:Real,
                        x::Array{T}  where T<:Real)
    n = length(y)
    # estimating m: propensity score
    model_m = StatsBase.fit(LassoPath, x, d, Binomial(), λ=10.0 .^ (-1:8))
    model_m_opt = selectmodel(model_m, MinCVmse(model_m, 10))
    # estimating g: E[Y | D, X]
    #model_g = DecisionTreeRegressor(max_depth=10, min_samples_leaf=500)
    #DecisionTree.fit!(model_g, [d x], y)
    model_g = _bagging_tree(50, y, d, x)
    return (model_m_opt, model_g)
end

"""
_bagging_tree(bs_reps::Int64, y::Array{T, 1} where T<:Real,
                       d::Array{T, 1} where T<:Real,
                       x::Array{T}  where T<:Real)

Bag regression trees.

##### Returns
- `tree_models`::Array{DecisionTreeRegressor,1} : array of regression trees
"""
function _bagging_tree(bs_reps::Int64, y::Array{T, 1} where T<:Real,
                       d::Array{T, 1} where T<:Real,
                       x::Array{T}  where T<:Real)
    n = length(y)
    tree_models = Array{DecisionTreeRegressor,1}(undef, bs_reps)
    for rep in 1:bs_reps
        bs_idx = rand(1:n, n)
        tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=500)
        DecisionTree.fit!(tree, [d[bs_idx] x[bs_idx, :]], y[bs_idx])
        tree_models[rep] = tree
    end
    return tree_models
end

"""
    _evaluate_m(model_m::GeneralizedLinearModel{
                                    GLM.GlmResp{Array{Float64,1},
                                    Binomial{Float64},LogitLink},
                                    GLM.DensePredQR{Float64}},
                     x::Array{T, 2} where T<:Real)

Evaluate ML model `model_m` at all rows of `x`.
"""
function _evaluate_m(model_m::GeneralizedLinearModel{
                                    GLM.GlmResp{Array{Float64,1},
                                    Binomial{Float64},LogitLink},
                                    GLM.DensePredQR{Float64}},
                     x::Array{T, 2} where T<:Real)
    n = size(x)[1]
    StatsBase.predict(model_m, [ones(n) x])
end

"""
_evaluate_g(model_g::Array{DecisionTreeRegressor,1},
                     d::Array{T, 1} where T<:Real,
                     x::Array{T, 2} where T<:Real)

Evaluate ML model `model_g` at all rows of `[d x]`.
"""
function _evaluate_g(model_g::Array{DecisionTreeRegressor,1},
                     d::Array{T, 1} where T<:Real,
                     x::Array{T, 2} where T<:Real)
    n = length(d)
    n_tree = length(model_g)
    predictions = zeros(n)
    for j in 1:n_tree
        predictions += DecisionTree.predict(model_g[j], [d x])
    end
    return predictions/n_tree
end


"""
    _theta_estimator(eta, y::Array{<:Real, 1},
                          d::Array{<:Real, 1},  x::Array{<:Real})

Estimate theta from Neyman orthogonality score.
"""
function _theta_estimator(eta, y::Array{<:Real, 1},
                          d::Array{<:Real, 1},  x::Array{<:Real})
    # objective function to be set equal to zero
    function psi!(F, theta)
        F[1] = mean(_neyman_score(eta, theta[1], y, d, x))
    end
    solver_result = nlsolve(psi!, [0.0])
    theta_hat = solver_result.zero[1]
    return theta_hat
end

"""
    _neyman_score(eta, theta::Float64,
                       y::Array{T, 1} where T<:Real,
                       d::Array{T, 1} where T<:Real,
                       x::Array{T, 2} where T<:Real)

Compute Neyman orthogonality score.

##### Returns
- `mean_score`::Float64 : mean Neyman orthogonality score over observations
"""
function _neyman_score(eta, theta::Float64,
                       y::Array{T, 1} where T<:Real,
                       d::Array{T, 1} where T<:Real,
                       x::Array{T, 2} where T<:Real)
    n = length(y)
    model_m, model_g = eta
    prop_score = _evaluate_m(model_m, x)
    g_t = _evaluate_g(model_g, ones(n), x)  # g(1, x)
    g_c = _evaluate_g(model_g, zeros(n), x)  #g(0, x)
    neyman_score = @. g_t - g_c + d * (y - g_t) / prop_score +
        (1.0 - prop_score) * (y - g_c) / (1 - prop_score) - theta
    #mean_score = sum(neyman_score) / n
    return neyman_score
end

"""
    _partition(N::Int64, K::Int64)
    
Partition the set {1,2,...,`N`} into `K` equal sized partitions.

##### Returns
- part::Array{Int64,1}:`N`-length array, with each entry in {1,2,...,K}
    indicating partition membership
"""
function _partition(N::Int64, K::Int64)
    ((N<1) || (K<1)) ? throw(DomainError("`N`, `K` must be > 1")) : nothing
    n = N%K == 0 ? Int(N / K) : throw(DomainError("`N` must be divisible by `K`"))
    part = repeat(1:K, n)
    shuffle!(part)
    return part
end

#df = read_data(); y = df.bweight; d = df.smoke; x = Array(df[:,Not([:bweight, :smoke])])
#x_poly = polynomial_features(x, 1)
#trees = _bagging_tree(20, y,d,x_poly)
#eta_hat = _eta_estimator(y,d,x_poly[:,2:end])
#model_m, model_g = eta_hat
#ns = _neyman_score(eta_hat, 0.0,y, d, x_poly[:,2:end])
#theta_hat = _theta_estimator(eta_hat, y, d, x_poly[:,2:end])
#(theta_hats, sigmasq_hats) = ate_estimator(y, d, x_poly[:, 2:end], 5)
