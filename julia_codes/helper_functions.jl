using Bijectors
using Random, ProgressBars
using Distributions, DistributionsAD
using NNlib, SpecialFunctions

"""
    invlink_parameters(θ::AbstractVector, K::Int, D::Int)

Map a vector of reals (θ) to the constrained space used for the log density problem.
The function splits the θ vector into the individual paramters and links them back to their constrained space if necessary.
"""
function invlink_parameters(θ::AbstractVector, K::Int, D::Int; pν = InverseGamma(2,3), pb_h = Uniform(-π, π))
    s = 1
    e = (s-1) + 1
    ν = invlink(pν, θ[s])

    s = e+1
    e = (s-1) + (K*D)
    w_h = reshape(θ[s:e], D, K)

    s = e+1
    e = (s-1) + (K)
    b_h = reshape(invlink.(pb_h, θ[s:e]), 1, K)

    s = e+1
    e = (s-1) + K
    w_o = θ[s:e]

    s = e+1
    b_o = θ[end]

    return (ν, w_h, b_h, w_o, b_o)
end

function invlink_parameters(θ::AbstractVector, K::Int; pν = InverseGamma(2,3), pb = Uniform(-π, π))
    s = 1
    e = (s-1) + 1
    ν = invlink(pν, θ[s])

    s = e+1
    e = (s-1) + K
    w_h = θ[s:e]

    s = e+1
    e = (s-1) + K
    b_h = reshape(invlink.(pb, θ[s:e]), 1, K)

    s = e+1
    e = (s-1) + K
    w_o = θ[s:e]

    return (ν, w_h, b_h, w_o)
end

"""
    init(K::Int, D::Int; seed = 2021)

Return initial parameters by randomly sampling them.
"""
function init(kernel::Kernel, K::Int, D::Int; seed = 2021)

    Random.seed!(seed)

    νℝ₊ = rand()
    w_h = init(kernel, D*K)
    bℝ₊_h = rand(K)
    w_o = randn(K)
    b_o = randn(K)

    θ = vcat(νℝ₊, w_h, bℝ₊_h, w_o, b_o)

    return θ
end

function init(K::Int; seed = 2021)

    Random.seed!(seed)

    νℝ₊ = rand()
    w_h = vec(randn(K))
    bℝ₊_h = vec(rand(K))
    w_o = randn(K)

    θ = vcat(νℝ₊, w_h, bℝ₊_h, w_o)

    return θ
end

# helper to compute pmf of Bernoulli
bernoulli(pj,yj) = yj == zero(yj) ? one(pj) - pj : pj

"""
    predict(Xnew, samples)

Return the mean function value for each Xnew.
"""
function predict_banana(Xnew, samples, kernel, K)

    Nnew, D = size(Xnew)
    pw_h, pw_o, pb_h, pb_o = getpriors(kernel, K, D)
    pν = InverseGamma(2,3) # prior on the lengthscale

    issincos = hasSinCosActivation(kernel)

    p = zeros(Nnew, length(samples))
    for (i, θ) in ProgressBar(enumerate(samples))
        ν, w_h, b_h, w_o, b_o = invlink_parameters(θ, K, D; pν = pν, pb_h = pb_h)

        z = issincos ? σ.(Ref(kernel), Xnew*(w_h*sqrt(ν))) : σ.(Ref(kernel), Xnew*(w_h*sqrt(ν)) .+ b_h)
        p[:,i] = logistic.(z * w_o .+ b_o)
    end

    return mean(p, dims=2), var(p, dims=2)
end

function predict_regression(xnew, samples, kernel, K)
    Nnew = length(xnew)
    pw_h, pw_o, pb = getpriors(kernel, K)
    pν = InverseGamma(2,3) # prior on the lengthscale

    p = zeros(Nnew, length(samples))
    for (i, θ) in enumerate(samples)
        ν, w_h, b_h, w_o = invlink_parameters(θ, K; pν = pν, pb = pb)
        z = σ.(Ref(kernel), xnew*(w_h') .+ b_h)
        p[:,i] = z * w_o
    end

    return mean(p, dims=2), var(p, dims=2)
end
