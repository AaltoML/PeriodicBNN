using Random, PGFPlotsX, KernelFunctions, Distributions, LaTeXStrings

X = reshape(collect(range(0.0,3.0,length=300)),:,1)

K = 5000

start = 25

sinactivation(x) = sqrt(2) * sin(x)
sincosactivation(x) = sin(x) + cos(x)

ffloor(x) = floor(x/π + 0.5)
_z(x) = (x - π * ffloor(x)) * (-1)^(ffloor(x))
triangeactivation(x) = π/(2*sqrt(2)) * _z(x)

_prelu(x) = π/4	* (_z(x) + _z(x + π/2))
prelu(x) = _prelu(x)

b = rand(-π:0.01:π, K)

f1 = sinactivation
f2 = sincosactivation

k1(x1, x2, w, f) = 1/K * sum( f(x1*w[k] + b[k]) * f(x2*w[k] + b[k]) for k in 1:K)
k2(x1, x2, w, f) = 1/K * sum( f(x1*w[k]) * f(x2*w[k]) for k in 1:K)

w₁ = rand(TDist(2 * 1/2), K)
k₁ = Matern12Kernel()
K₁ = kernelmatrix(k₁,X,obsdim=1)

w₂ = rand(TDist(2 * 3/2), K)
k₂ = Matern32Kernel()
K₂ = kernelmatrix(k₂,X,obsdim=1)

w₃ = rand(Normal(), K)
k₃ = SqExponentialKernel()
K₃ = kernelmatrix(k₃,X,obsdim=1)

z_sin₁ = k1.(Ref(X[start]), X, Ref(w₁), Ref(sinactivation))
z_sin₂ = k1.(Ref(X[start]), X, Ref(w₂), Ref(sinactivation))
z_sin₃ = k1.(Ref(X[start]), X, Ref(w₃), Ref(sinactivation))

z_sincos₁ = k2.(Ref(X[start]), X, Ref(w₁), Ref(sincosactivation))
z_sincos₂ = k2.(Ref(X[start]), X, Ref(w₂), Ref(sincosactivation))
z_sincos₃ = k2.(Ref(X[start]), X, Ref(w₃), Ref(sincosactivation))

z_tri₁ = k1.(Ref(X[start]), X, Ref(w₁), Ref(triangeactivation))
z_tri₂ = k1.(Ref(X[start]), X, Ref(w₂), Ref(triangeactivation))
z_tri₃ = k1.(Ref(X[start]), X, Ref(w₃), Ref(triangeactivation))

z_relu₁ = k1.(Ref(X[start]), X, Ref(w₁), Ref(prelu))
z_relu₂ = k1.(Ref(X[start]), X, Ref(w₂), Ref(prelu))
z_relu₃ = k1.(Ref(X[start]), X, Ref(w₃), Ref(prelu))


@pgf p1 = Axis(
    {
        no_markers,
        xlabel = raw"Input, $|x-x'|$",
        width = "15cm",
        height = "10cm",
        ymax = 1.2,
        no_markers,
        xticklabels={},
        yticklabels={},
        ymin = 0,
    },
    # Exponential
    PlotInc({color = "black", dashed}, Table(1:300, K₁[start,:])),
    LegendEntry("Exact"),

    PlotInc({color = "red", solid}, Table(1:300, vec(z_sin₁))),
    LegendEntry("Sinusiod"),

    PlotInc({color = "green", solid}, Table(1:300, vec(z_sincos₁))),
    LegendEntry("Sine Cosine"),

    # Matern 3/2
    PlotInc({color = "black", dashed}, Table(201:500, K₂[start,:])),
    PlotInc({color = "red", solid}, Table(201:500, vec(z_sin₂))),
    PlotInc({color = "green", solid}, Table(201:500, vec(z_sincos₂))),

    # Squared Exponential
    PlotInc({color = "black", dashed}, Table(401:700, K₃[start,:])),
    PlotInc({color = "red", solid}, Table(401:700, vec(z_sin₃))),
    PlotInc({color = "green", solid}, Table(401:700, vec(z_sincos₃))),

    [raw"\node at ",
     Coordinate(25, 1.1),
     raw"{Exponential};"],
     [raw"\node at ",
     Coordinate(225, 1.1),
     raw"{Mat\'ern-$\frac{3}{2}$};"],
     [raw"\node at ",
     Coordinate(425, 1.1),
     raw"{Squared Exponential};"]
)

@pgf p2 = Axis(
    {
        no_markers,
        xlabel = raw"Input, $|x-x'|$",
        width = "15cm",
        height = "10cm",
        ymax = 1.2,
        no_markers,
        xticklabels={},
        yticklabels={},
        ymin = 0,
    },
    # Exponential
    PlotInc({color = "black", dashed}, Table(1:300, K₁[start,:])),
    LegendEntry("Exact"),

    PlotInc({color = "blue", solid}, Table(1:300, vec(z_tri₁))),
    LegendEntry("Triangle Wave"),

    PlotInc({color = "orange", solid}, Table(1:300, vec(z_relu₁))),
    LegendEntry("Periodic ReLU"),

    # Matern 3/2
    PlotInc({color = "black", dashed}, Table(201:500, K₂[start,:])),
    PlotInc({color = "blue", solid}, Table(201:500, vec(z_tri₂))),
    PlotInc({color = "orange", solid}, Table(201:500, vec(z_relu₂))),

    # Squared Exponential
    PlotInc({color = "black", dashed}, Table(401:700, K₃[start,:])),
    PlotInc({color = "blue", solid}, Table(401:700, vec(z_tri₃))),
    PlotInc({color = "orange", solid}, Table(401:700, vec(z_relu₃))),

    [raw"\node at ",
     Coordinate(25, 1.1),
     raw"{Exponential};"],
     [raw"\node at ",
     Coordinate(225, 1.1),
     raw"{Mat\'ern-$\frac{3}{2}$};"],
     [raw"\node at ",
     Coordinate(425, 1.1),
     raw"{Squared Exponential};"]
)

@pgf gp = GroupPlot(
    {
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = "0.2cm",
        },
        no_markers,
        xticklabels={},
        yticklabels={},
        ymin = 0,
    },
    p1,
    p2,
)

pgfsave("kernels_p1.tex", p1, include_preamble = false)
pgfsave("kernels_p2.tex", p2, include_preamble = false)

pgfsave("kernels.pdf", gp)
pgfsave("kernels.tex", gp)

x = range(-2*pi; stop = 2*pi, length = 500)

@pgf sin_plot = Axis(
    {
        ylabel = raw"$\sigma(x)$",
        no_markers,
        ymin = -2.5,
        ymax = 2.5,
        xticklabels={},
    },
    Plot({color = "red"},
    Table(x, sinactivation.(x))
    ), LegendEntry("Sinusiod")
    )
@pgf sincos_plot = Axis(
    {
        no_markers,
        ymin = -2.5,
        ymax = 2.5,
        xticklabels={},
    },
    PlotInc({color = "green"}, Table(x, sincosactivation.(x))), LegendEntry("Sine Cosine"),
    PlotInc({dashed, color = "red"}, Table(x, sinactivation.(x)))
    )
@pgf tri_plot = Axis(
    {
        no_markers,
        ymin = -2.5,
        ymax = 2.5,
        xticklabels={},
    },
    PlotInc({color = "blue"}, Table(x, triangeactivation.(x))), LegendEntry("Triangle Wave"))
@pgf relu_plot = Axis(
    {
        no_markers,
        ymin = -2.5,
        ymax = 2.5,
        xticklabels={},
    },
    PlotInc({color = "orange"}, Table(x, prelu.(x))), LegendEntry("Periodic ReLU"))

@pgf gp = GroupPlot(
    {
        group_style = {
            group_size = "4 by 1",
            yticklabels_at="edge left",
            xticklabels_at="edge bottom",
            horizontal_sep = "0.2cm",
        },
        no_markers,
        ymin = -2.5,
        ymax = 2.5,
        xticklabels={},
    },
    sin_plot,
    sincos_plot,
    tri_plot,
    relu_plot
)

pgfsave("sin_activation.tex", sin_plot, include_preamble = false)
pgfsave("sincosactivation.tex", sincos_plot, include_preamble = false)
pgfsave("triangle_activation.tex", tri_plot, include_preamble = false)
pgfsave("relu_activation.tex", relu_plot, include_preamble = false)

pgfsave("activations.pdf", gp)
pgfsave("activations.tex", gp)
