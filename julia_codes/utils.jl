function getkernel(parsed_args)

    afun = if parsed_args["activation"] == "triangle"
        TriangleWave()
    elseif parsed_args["activation"] == "prelu"
        PeriodicReLU()
    elseif parsed_args["activation"] == "sincos"
        SinCosActivation()
    else
        SinActivation()
    end

    if parsed_args["kernel"] == "Matern"
        return Matern(afun,parsed_args["nu"])
    elseif parsed_args["kernel"] == "RelU"
        return RELU()
    elseif parsed_args["kernel"] == "MaternLS"
        return MaternLS(parsed_args["nu"], parsed_args["ell"])
    elseif parsed_args["kernel"] == "RBFLS"
        return RBFLS()
    else
        return RBF(afun)
    end
end
