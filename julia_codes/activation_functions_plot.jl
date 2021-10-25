using Random, Turing, PGFPlotsX, NNlib, SpecialFunctions, LaTeXStrings
using StatsFuns: logistic, norminvcdf

using Memoization
using ReverseDiff

using Zygote

N = 50
X = vcat(-rand(N) .- 0.5, rand(N) .+ 0.5)
l = vcat(ones(N), zeros(N))
y = l + 0.1*randn(2*N)

xt = range(-5., 5., length = 100);

const triangleconst = π/(2*sqrt(2))
const preluconst = π/4
_trianglewave(x) = (x - π * floor(x/π + 0.5)) * (-1)^floor(x/π + 0.5)
_trianglewave(x::ReverseDiff.TrackedReal) = ReverseDiff.ForwardOptimize(_trianglewave)(x)

heaviside(x) = x < zero(x) ? zero(x) : x >= zero(x) ? one(x) : oftype(x,0.5)
lsmatern(x, q, ν, λ) = q*heaviside(x)*x^(ν-1/2)*exp(-λ*x)
heaviside(x::ReverseDiff.TrackedReal) = ReverseDiff.ForwardOptimize(heaviside)(x)

abstract type PeriodicFunction end

struct SinActivation <: PeriodicFunction end
const sineconst = sqrt(2)
(σ::SinActivation)(x::T) where T = T(sineconst) * sin(x)

struct SinCosActivation <: PeriodicFunction end
(σ::SinCosActivation)(x) = sin(x) + cos(x)

struct TriangleWave <: PeriodicFunction end
(σ::TriangleWave)(x) = triangleconst * _trianglewave(x)

struct PeriodicReLU <: PeriodicFunction end
(σ::PeriodicReLU)(x) = preluconst * (_trianglewave(x) + _trianglewave(x + π/2))

abstract type ActivationFunction end

struct ReLU <: ActivationFunction end
(σ::ReLU)(x) = relu(x)
prior_weights(::ReLU) = Normal()
prior_bias(::ReLU) = Normal()
getname(::ReLU) = "relu"

struct Step <: ActivationFunction end
(σ::Step)(x) = heaviside(x)
prior_weights(::Step) = Normal()
prior_bias(::Step) = Normal()
getname(::Step) = "step"

struct Sigmoid <: ActivationFunction end
(σ::Sigmoid)(x) = sigmoid(x)
prior_weights(::Sigmoid) = Normal()
prior_bias(::Sigmoid) = Normal()
getname(::Sigmoid) = "sigmoid"

struct RBFNN <: ActivationFunction end
(σ::RBFNN)(x) = exp(-x^2)
prior_weights(::RBFNN) = Normal()
prior_bias(::RBFNN) = Normal()
getname(::RBFNN) = "RBF_NN"

struct MaternNN <: ActivationFunction
    q # precomputed constant
    λ # precomputed constant
    ν # smoothness
    function MaternNN(ν)
        λf(ν) = sqrt(2*ν)
        Af(ν) = sqrt(2 * π^(1/2) * λf(ν)^(2*ν) * gamma(ν + 0.5) / gamma(ν)) / gamma(ν + 0.5)
        return new(Af(ν), λf(ν), ν)
    end
end
(σ::MaternNN)(x) = σ.q*heaviside(x)*x^(σ.ν-1/2)*exp(-σ.λ*x)
prior_weights(::MaternNN) = Normal()
prior_bias(::MaternNN) = Normal()
getname(σ::MaternNN) = "Matern_NN_$(Int(σ.ν*10))"

struct RBF <: ActivationFunction
    σ::PeriodicFunction
end
(σ::RBF)(x) = σ.σ(x)
prior_weights(::RBF) = Normal()
prior_bias(::RBF) = Uniform(-π, π)
getname(::RBF) = "RBF"

struct Matern <: ActivationFunction
    σ::PeriodicFunction
    ν # smoothness
end
(σ::Matern)(x) = σ.σ(x)
prior_weights(σ::Matern) = σ.ν == 0.5 ? Cauchy() : TDist(2*σ.ν)
prior_bias(::Matern) = Uniform(-π, π)
getname(σ::Matern) = "Matern-$(Int(σ.ν*10))"

@pgf p = Axis(
    { 
        ylabel= raw"$\sigma(x)$",
        no_markers,
        xticklabels={},
        yticklabels={},
        width = "15cm",
    }, 
    PlotInc(Table(xt, RBF(SinActivation()).(xt))),
    LegendEntry("sinusiod"),
    PlotInc(Table(xt, RBF(SinCosActivation()).(xt))),
    LegendEntry("sine-cosine"),
    PlotInc(Table(xt, RBF(TriangleWave()).(xt))),
    LegendEntry("triangle wave"),
    PlotInc(Table(xt, RBF(PeriodicReLU()).(xt))),
    LegendEntry("periodic ReLU"),
)

display("image/svg+xml", p)

@model function bnn(x, y, afun, K)
    
    w ~ filldist(prior_weights(afun), 1, K)
    b ~ filldist(prior_bias(afun), 1, K)
    
    l ~ FlatPos(0)
    z = afun.(l * x * w .+ b)
    
    wo ~ filldist(Normal(), K)
    bo ~ Normal()
    
    f = (z * (wo / K) .+ bo) / 2
    
    s ~ FlatPos(0)
    Turing.@addlogprob! -sum(abs2((yj-fj)/s)/2 + log(s) for (fj,yj) in zip(f,y))
end

Turing.setadbackend(:reversediff)
Turing.setrdcache(true);
Turing.setprogress!(true);

function predict(x, afun, w::Chains, b::Chains, l, wo::Chains, bo, s; burnin = 5000)
    nsamples = length(w)
    
    latents = map(s -> begin
            w_ = Array(w[s])
            b_ = Array(b[s])
            
            l_ = l[s]
            
            wo_ = vec(Array(wo[s]))
            bo_ = bo[s]
            
            z = afun.(x * l_ * w_ .+ b_)
            f = z * (wo_ / length(wo_)) .+ bo_
            f ./= 2
        end, burnin:nsamples)
    
    err_o = 1.96*sqrt.(var(latents)) .+ s
    err = 1.96*std(latents)
    
    return mean(latents), err, err_o
end

function plot(m, sf, so)

    @pgf p = Axis(
        {
            "scatter/classes" = {
                0 = {"black", mark_size="0.5"},
                1 = {"black", mark_size="0.5"},
            },
            axis_lines = "none",
            ymin = minimum(m .- so) - 0.25,
            ymax = maximum(m .+ so) + 0.25
        },
        PlotInc( { no_markers, white, solid }, Table(xt, m) ),
        PlotInc( { no_markers, white, solid }, Table(xt, m .+ so) ),
        PlotInc( { no_markers, white, solid }, Table(xt, m .- so) ),

        PlotInc( { scatter, "only marks", "scatter src" = "explicit symbolic", },
            Table({ meta = "label" }, x = X, y = y, label = Int.(l)) )
        )
    
    return p
end

function plot_lines(m, sf, so, filename)
    
        @pgf p_lines = Axis(
        {
            "scatter/classes" = {
                0 = {"black", mark_size="0.1"},
                1 = {"black", mark_size="0.1"},
            },
            ymin = minimum(m .- so) - 0.2,
            ymax = maximum(m .+ so) + 0.2
        },
        
        PlotInc( Graphics({xmin=minimum(xt),xmax=maximum(xt), ymin=minimum(m .- so), ymax=maximum(m .+ so)}, "$(pwd())/$(filename)") ),
        PlotInc( { no_markers, gray, dotted }, Table(xt, ones(length(xt))*0.5)),
        PlotInc( { no_markers, red, solid }, Table(xt, m) ),
        PlotInc( { no_markers, black, dashed }, Table(xt, m .+ sf) ),
        PlotInc( { no_markers, black, dashed }, Table(xt, m .- sf) ),
        PlotInc( { no_markers, black, solid }, Table(xt, m .+ so) ),
        PlotInc( { no_markers, black, solid }, Table(xt, m .- so) ),
        )
    
    return p_lines
end

afuns = [
    ReLU(), 
    Step(), 
    Sigmoid(), 
    Matern(SinActivation(), 1/2), 
    Matern(SinActivation(), 3/2), 
    RBF(SinActivation()),
    MaternNN(1/2, 1),
    MaternNN(3/2, 1),
    RBFNN()
]

K = 10
for afun in afuns#[k:k]
    
    @info afun

    model = bnn(X, y, afun, K);
    chain = sample(model, NUTS(1000, 0.75), 5000);
    
    @info "Finished sampling"

    # extract names

    w_names = filter(n -> contains(string(n), "w["), names(chain))
    wo_names = filter(n -> contains(string(n), "wo["), names(chain))
    b_names = filter(n -> contains(string(n), "b["), names(chain))

    m, sf, so = predict(xt, afun, 
        chain[w_names], 
        chain[b_names], 
        chain["l"], 
        chain[wo_names], 
        chain["bo"], 
        mean(chain["s"]),
        burnin = 1000
    )
    
    p = plot(m, sf, so)
    
    @info "Saving plots"
    
    pgfsave("$(getname(afun))_plot.png", p)
    
    p = plot_lines(m, sf, so, "$(getname(afun))_plot.png")
    
    pgfsave("$(getname(afun))_plot.pdf", p)
    pgfsave("$(getname(afun))_plot.tex", p, include_preamble = false)
end
