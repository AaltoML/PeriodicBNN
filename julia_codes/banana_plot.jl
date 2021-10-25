using ArgParse
using PyPlot, NNlib, StatsFuns
using Serialization, DelimitedFiles

"""
    parse_commandline()

This function parses the arguments of the call `julia banana.jl`.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin

        "datapath"
            help = "dataset path"
            required = true

        "resultspath"
            help = "results path"
            required = true

        "uid"
            help = "uid"
            required = true

        "kernelstring"
            help = "kernel string"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

datapath = parsed_args["datapath"]
datadir = parsed_args["resultspath"]
uid = parsed_args["uid"]
kernel = parsed_args["kernelstring"]

include("functions.jl")
include("helper_functions.jl")
include("utils.jl")

samples = deserialize(joinpath(datadir, "banana_samples_$(uid)_$(kernel).jd"))
stats = deserialize(joinpath(datadir, "banana_stats_$(uid)_$(kernel).jd"))
K, D, n_samples, n_adapts, ad, kernel, seed, pν = deserialize(joinpath(datadir, "banana_setup_$(uid)_$(kernel).jd"))


X = readdlm(joinpath(datapath, "banana_datapoints_subset.csv"), ',')
y = vec(Int.(readdlm(joinpath(datapath, "banana_classes_subset.csv"), ',')))

gridlength = 3.75
xmin, ymin = -gridlength, -gridlength
xmax, ymax = gridlength, gridlength

grid = mapreduce(yi -> mapreduce(xi -> [xi, yi], hcat, range(xmin, xmax, length=300)), hcat, range(ymin, ymax, length=300))'
yhat, σhat = predict_banana(grid, samples, kernel, K)

fig1, ax1 = plt.subplots()

ax1.imshow(reshape(1 .- yhat, 300, 300)', cmap="RdBu", origin="lower", vmin = 0, vmax = 1, extent=[-gridlength, gridlength, -gridlength, gridlength])
ax1.plot(X[y .== false,1], X[y .== false,2], "o", markersize=6.5, markerfacecolor="white", markeredgewidth=1.5, markeredgecolor="blue", alpha=1.0)
ax1.plot(X[y .== true,1], X[y .== true,2], "v", markersize=6.5, markerfacecolor="white", markeredgewidth=1.5, markeredgecolor="red", alpha=1.0)

ax1.axis("off")

fig1.tight_layout()
savefig("banana_$(kernel)_expectation.pdf", bbox_inches = "tight", pad_inches = 0)

display(gcf())
