using ArgParse
using AdvancedHMC
using StatsFuns, Distributions
using DistributionsAD, VectorizedRoutines
using ForwardDiff, Zygote
using DelimitedFiles, Random, LinearAlgebra, Serialization
using NNlib

BLAS.set_num_threads(8)

using ReverseDiff, Memoization
using ReverseDiff: compile, GradientTape
using ReverseDiff.DiffResults: GradientResult

# load utility functions
include("functions.jl")
include("helper_functions.jl")
include("utils.jl")

"""
    parse_commandline()

This function parses the arguments of the call `julia banana.jl`.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--nsamples"
            help = "Number of MCMC samples"
            arg_type = Int
            default = 1000
        "--nadapts"
            help = "Number of adaptation samples"
            arg_type = Int
            default = 1000
        "--K"
            help = "Number of hidden units"
            arg_type = Int
            default = 10
        "--kernel"
            help = "Kernel: [RBF, Matern, RBFLS, MaternLS, ReLU]"
            arg_type = String
            default = "RBF"
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 2021
        "--nu"
            help = "Matern parameter"
            arg_type = Float64
            default = 3/2
        "--ell"
            help = "Matern parameter"
            arg_type = Float64
            default = 1.0
        "--ad"
            help = "AD mode (reverse, forward or zygote (experimental))"
            arg_type = String
            default = "forward"
        "--activation"
            help = "Periodic activation function: [sin, sincos, triangle, prelu]"
            arg_type = String
            default = "sin"
        "--hideprogress"
            help = "Hide progress"
            action = :store_true
        "--subsample"
            help = "amount of training samples used (in %), minimum 10%."
            arg_type = Int
            default = 50
        "--subsampleseed"
            help = "Random seed for subsampling"
            arg_type = Int
            default = 2021

        "datapath"
            help = "dataset path"
            required = true

        "outputpath"
            help = "output path"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

datapath = parsed_args["datapath"]
outputpath = parsed_args["outputpath"]

# Random seed to be used
seed = parsed_args["seed"]

# unique run id
uid = rand(Int)
uid *= sign(uid)

# load dataset
X = readdlm(joinpath(datapath, "banana_datapoints.csv"), ',')
y = vec(Bool.(readdlm(joinpath(datapath, "banana_classes.csv"), ',')))

subsample = max(parsed_args["subsample"], 10)

# subsample the data
if subsample < 100
    N = size(X,1)
    ids = collect(1:N)
    Random.seed!(parsed_args["subsampleseed"])
    shuffle!(ids)

    M = round(Int, N * (subsample / 100))

    X = X[ids[1:M],:]
    y = y[ids[1:M]]
end

N, D = size(X)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = parsed_args["nsamples"], parsed_args["nadapts"]

K = parsed_args["K"]

# set the kernel
kernel = getkernel(parsed_args)

# priors used for inference
pν = InverseGamma(2,3) # prior on the lengthscale
pw_h, pw_o, pb_h, pb_o = getpriors(kernel, K, D) # priors on the weights and biases

function _logjoint_biasfree(θ)

    ν, w_h, _, w_o, b_o = invlink_parameters(θ, K, D; pν = pν, pb_h = pb_h)

    lp = 0.0

    # priors
    lp += logpdf_with_trans(pν, ν, true)
    lp += logpdf(pw_h, vec(w_h))
    lp += logpdf(pw_o, w_o)
    lp += logpdf(pb_o, b_o)

    zval = X*(w_h*sqrt(ν))

    p = logistic.(σ.(Ref(kernel), zval) * w_o .+ b_o)

    # likelihood
    lp += sum(log(bernoulli(pj,yj)) for (pj,yj) in zip(p,y))

    return lp
end

function _logjoint(θ)

    ν, w_h, b_h, w_o, b_o = invlink_parameters(θ, K, D; pν = pν, pb_h = pb_h)

    lp = 0.0

    # priors
    lp += logpdf_with_trans(pν, ν, true)
    lp += logpdf(pw_h, vec(w_h))
    lp += sum(logpdf_with_trans(pb_h, vec(b_h), true))
    lp += logpdf(pw_o, w_o)
    lp += logpdf(pb_o, b_o)

    zval = X*(w_h*sqrt(ν)) .+ b_h

    p = logistic.(σ.(Ref(kernel), zval) * w_o .+ b_o)

    # likelihood
    lp += sum(log(bernoulli(pj,yj)) for (pj,yj) in zip(p,y))

    return lp
end


logjoint = hasSinCosActivation(kernel) ? _logjoint_biasfree : _logjoint

# construct initial parameters by random sampling
θinit = init(kernel, K, D; seed = seed)

f_tape = GradientTape(logjoint, θinit)
compiled_f_tape = compile(f_tape)

function gradient_logjoint(θ)
    result = GradientResult(θ)
    ReverseDiff.gradient!(result, compiled_f_tape, θ)
    l = DiffResults.value(result)
    ∂l∂θ::typeof(θ) = DiffResults.gradient(result)
    return l, ∂l∂θ
end

logjoint(θinit)
gradient_logjoint(θinit)

# --
# End of ReverseDiff specific code.
# --

# set the AD that should be used (ForwardDiff (forward-mode AD) or ReverseDiff (reverse-mode AD))
ad = parsed_args["ad"] == "forward" ? ForwardDiff : parsed_args["ad"] == "reverse" ? gradient_logjoint : parsed_args["ad"] == "zygote" ? Zygote : ForwardDiff

# Define a Hamiltonian system
metric = DiagEuclideanMetric(length(θinit))
hamiltonian = Hamiltonian(metric, logjoint, ad)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, θinit)
integrator = JitteredLeapfrog(initial_ϵ, 0.1)

# Define an HMC sampler (we use the STAN equvalient here)
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

kernelstr = string(kernel)

@info "Running inference with the following setting:"
@show K, n_samples, n_adapts, kernelstr, ad, seed, datapath, outputpath

# Run the sampler
samples, stats = sample(hamiltonian, proposal, θinit, n_samples, adaptor, n_adapts; progress=!parsed_args["hideprogress"])

@info "Storing results with UID = $uid and kernel string = $kernelstr to: banana_samples_$(uid)_$(kernelstr).jd, banana_stats_$(uid)_$(kernelstr).jd and banana_setup_$(uid)_$(kernelstr).jd"

# save samples & stats & settings
serialize(joinpath(outputpath, "banana_samples_$(uid)_$(kernelstr).jd"), samples)
serialize(joinpath(outputpath, "banana_stats_$(uid)_$(kernelstr).jd"), stats)
serialize(joinpath(outputpath, "banana_setup_$(uid)_$(kernelstr).jd"), (K, D, n_samples, n_adapts, parsed_args["ad"], kernel, seed, pν))
