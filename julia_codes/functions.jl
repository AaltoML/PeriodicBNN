using Random
using Distributions, DistributionsAD
using NNlib, SpecialFunctions
using ReverseDiff

const triangleconst = π/(2*sqrt(2))
const preluconst = π/4
_trianglewave(x) = (x - π * floor(x/π + 0.5)) * (-1)^floor(x/π + 0.5)
_trianglewave(x::ReverseDiff.TrackedReal) = ReverseDiff.ForwardOptimize(_trianglewave)(x)

# Helper types for the activation functions
abstract type ActivationFunction end
struct SinActivation <: ActivationFunction end
const sineconst = sqrt(2)
(σ::SinActivation)(x::T) where T = T(sineconst) * sin(x)

struct SinCosActivation <: ActivationFunction end
(σ::SinCosActivation)(x) = sin(x) + cos(x)

struct TriangleWave <: ActivationFunction end
(σ::TriangleWave)(x) = triangleconst * _trianglewave(x)

struct PeriodicReLU <: ActivationFunction end
(σ::PeriodicReLU)(x) = preluconst * (_trianglewave(x) + _trianglewave(x + π/2))

# Helper type for different kernel function and their respective priors
abstract type Kernel end

hasSinCosActivation(::Kernel) = false

struct RBF{F<:ActivationFunction} <: Kernel
    σ::F
end

hasSinCosActivation(::RBF{SinCosActivation}) = true

getpriors(::RBF, K, D) = (MvNormal(zeros(K*D), 1), MvNormal(zeros(K), 1), DistributionsAD.TuringUniform(-π, π), Normal())
getpriors(::RBF, K) = (MvNormal(zeros(K), 1), MvNormal(zeros(K), 1), DistributionsAD.TuringUniform(-π, π), Normal())
init(::RBF, K) = randn(K)
Base.string(::RBF{T}) where {T} = "RBF_$(T)"

# Williams (1997)
struct RBFLS <: Kernel end
getpriors(::RBFLS, K, D) = (MvNormal(zeros(K*D), 1), MvNormal(zeros(K), 1), Normal(), Normal())
getpriors(::RBFLS, K) = (MvNormal(zeros(K), 1), MvNormal(zeros(K), 1), Normal(), Normal())
init(::RBFLS, K) = randn(K)
Base.string(::RBFLS) = "RBFLS"

struct Matern{F<:ActivationFunction,T<:AbstractFloat} <: Kernel
    σ::F
    ν::T
end
getpriors(kernel::Matern, K, D) = (filldist(TDist(2*kernel.ν), K*D), MvNormal(zeros(K), 1), DistributionsAD.TuringUniform(-π, π), Normal())
getpriors(kernel::Matern, K) = (filldist(TDist(2*kernel.ν), K), MvNormal(zeros(K), 1), DistributionsAD.TuringUniform(-π, π), Normal())
init(kernel::Matern, K) = rand(TDist(2*kernel.ν), K)
Base.string(m::Matern{T,V}) where {T,V} = "Matern_$(T)_$(Int(m.ν*10))"

hasSinCosActivation(::Matern{SinCosActivation,T}) where T = true

# Meronen (2020)
struct MaternLS{T<:AbstractFloat} <: Kernel
    q::T
    λ::T
    ν::T
end
function MaternLS(ν, ℓ)
    λf(ℓ, ν) = sqrt(2*ν)/ℓ
    Af(ℓ, ν) = sqrt(2 * π^(1/2) * λf(ℓ,ν)^(2*ν) * gamma(ν + 0.5) / gamma(ν)) / gamma(ν + 0.5)

    return MaternLS(Af(ℓ,ν), λf(ℓ,ν), ν)
end

getpriors(::MaternLS, K, D) = (MvNormal(zeros(K*D), 1), MvNormal(zeros(K), 1), Normal(), Normal())
getpriors(::MaternLS, K) = (MvNormal(zeros(K), 1), MvNormal(zeros(K), 1), Normal(), Normal())
init(::MaternLS, K) = randn(K)
Base.string(m::MaternLS{V}) where {V} = "MaternLS_$(Int(m.ν*10))"

struct RELU <: Kernel end
getpriors(::RELU, K, D) = (MvNormal(zeros(K*D), 1), MvNormal(zeros(K), 1), Normal(), Normal())
getpriors(::RELU, K) = (MvNormal(zeros(K), 1), MvNormal(zeros(K), 1), Normal(), Normal())
init(::RELU, K) = randn(K)
Base.string(::RELU) = "RELU"

σ(k::RBF, x) = k.σ(x)
σ(k::Matern, x) = k.σ(x)
σ(::RELU, x) = relu(x)
heaviside(x) = x < zero(x) ? zero(x) : x > zero(x) ? one(x) : oftype(x,0.5)
lsmatern(x, q, ν, λ) = q*heaviside(x)*x^(ν-1/2)*exp(-λ*x)
heaviside(x::ReverseDiff.TrackedReal) = ReverseDiff.ForwardOptimize(heaviside)(x)
σ(k::MaternLS, x) = lsmatern(x, k.q, k.ν, k.λ)
σ(::RBFLS, x) = exp(-x^2)
