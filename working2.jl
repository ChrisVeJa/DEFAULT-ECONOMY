###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random,
    Distributions,
    Statistics,
    LinearAlgebra,
    StatsBase,
    Parameters,
    Flux,
    ColorSchemes,
    Gadfly,
    Cairo,
    Fontconfig,
    Tables,
    DataFrames,
    Compose
include("supcodes.jl");

############################################################
# [1]  FUNCTIONS TO BE USED
############################################################
# ----------------------------------------------------------
# [1.a] Choosing unique states from the simulated data
# ----------------------------------------------------------
myunique(data) = begin
    dataTu = [Tuple(data[i, :]) for i = 1:size(data)[1]]
    dataTu = unique(dataTu)
    dataTu = [[dataTu[i]...]' for i = 1:length(dataTu)]
    data = [vcat(dataTu...)][1]
    return data
end
# ----------------------------------------------------------
# [1.b]  Normalization:
#     ̃x = (x - 1/2(xₘₐₓ + xₘᵢₙ)) /(1/2(xₘₐₓ - xₘᵢₙ))
# ----------------------------------------------------------
mynorm(x) = begin
    ux = maximum(x, dims = 1)
    lx = minimum(x, dims = 1)
    nx = (x .- 0.5 * (ux + lx)) ./ (0.5 * (ux - lx))
    return nx, ux, lx
end
# ----------------------------------------------------------
# [1.c]  Training of neural network
# ----------------------------------------------------------
    mytrain(NN,data) = begin
        lossf(x,y) = Flux.mse(NN(x),y);
        pstrain = Flux.params(NN)
        Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, data, Descent())
    end
# ----------------------------------------------------------
# [1.d]  Policy function conditional on expected values
# ----------------------------------------------------------
    update_solve(hat_vr, hat_vd, settings,params,uf) = begin
        # Starting the psolution of the model
        @unpack P, b,y = settings
        @unpack r, β, ne, nx, σrisk,θ = params
        p0 = findmin(abs.(0 .- b))[2]
        udef = repeat(settings.udef', ne, 1)
        hat_vf = max.(hat_vr,hat_vd)
        hat_D  = 1 * (hat_vd .> hat_vr)
        evf1 = hat_vf * P'
        evd1 = hat_vd * P'
        eδD1 = hat_D  * P'
        q1   = (1 / (1 + r)) * (1 .- eδD1) # price
        qb1  = q1 .* b
        βevf1= β*evf1
        vrnew = Array{Float64,2}(undef,ne,nx)
        cc1    = Array{Float64,2}(undef,ne,nx)
        bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
        yb    = b .+ y'
        @inbounds for i = 1:ne
            cc1 = yb[i, :]' .- qb1
            cc1 = max.(cc1,0)
            aux_u = uf.(cc1, σrisk) + βevf1
            vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
        end
        bb    = repeat(b, 1, nx)
        bb    = bb[bpnew]
        evaux = θ * evf1[p0, :]' .+  (1 - θ) * evd1
        vdnew = udef + β*evaux
        vfnew = max.(vrnew, vdnew)
        Dnew  = 1 * (vdnew .> vrnew)
        eδD   = Dnew  * P'
        qnew  = (1 / (1 + r)) * (1 .- eδD)
        return (vf = vfnew, vr = vrnew, vd = vdnew, D = Dnew, bb =  bb, q = qnew, bp = bpnew)
    end

############################################################
# [2] SETTING
############################################################
params = (
    r = 0.017,
    σrisk = 2.0,
    ρ = 0.945,
    η = 0.025,
    β = 0.953,
    θ = 0.282,
    nx = 21,
    m = 3,
    μ = 0.0,
    fhat = 0.969,
    ub = 0,
    lb = -0.4,
    tol = 1e-8,
    maxite = 500,
    ne = 251,
);
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
polfun, settings = Solver(params, hf, uf);
MoDel = [vec(polfun[i]) for i = 1:6]
MoDel =
    [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1)) hcat(MoDel...)]
heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q]
ModelData = DataFrame(Tables.table(MoDel, header = heads))
# ----------------------------------------------------------
# [3.b] Plotting results from the Model
# ----------------------------------------------------------
set_default_plot_size(18cm, 12cm)
p0 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "vf",
    color = "output",
    Geom.line,
    Theme(
        background_color = "white",
        key_position = :right,
        key_title_font_size = 6pt,
        key_label_font_size = 6pt,
    ),
    Guide.ylabel("Value function"),
    Guide.xlabel("Debt (t)"),
    Guide.title("Value function by output level"),
)
draw(PNG("./Plots/ValuFunction.png"), p0);

set_default_plot_size(12cm, 8cm)
p1 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "vr",
    color = "output",
    Geom.line,
    Theme(
        background_color = "white",
        key_position = :right,
        key_title_font_size = 6pt,
        key_label_font_size = 6pt,
    ),
    Guide.ylabel("Value of repayment", orientation = :vertical),
    Guide.xlabel("Debt (t)"),
    Guide.title("(a) Value function for Repayment"),
);

p2 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "vd",
    color = "output",
    Geom.line,
    Theme(background_color = "white", key_position = :none),
    Guide.ylabel("Value of Default", orientation = :vertical),
    Guide.xlabel("Debt (t)"),
    Guide.title("(b) Value function for Default"),
);

p3 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "b",
    color = "output",
    Geom.line,
    Theme(background_color = "white", key_position = :none),
    Guide.ylabel("Debt policy (t+1)", orientation = :vertical),
    Guide.xlabel("Debt (t)"),
    Guide.title("(c) Debt policy function"),
);


ytick = round.(settings.y, digits = 2)
yticks = [ytick[1], ytick[6], ytick[11], ytick[16], ytick[end]]
p4 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "output",
    color = "D",
    Geom.rectbin,
    Scale.color_discrete_manual("yellow", "black"),
    Theme(background_color = "white", key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"),
    Guide.xlabel("Debt (t)"),
    Guide.colorkey(title = "Default choice", labels = ["Default", "No Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]),
    Guide.yticks(ticks = yticks),
    Guide.title("(d) Default Choice"),
);
set_default_plot_size(18cm, 12cm)
h0 = Gadfly.gridstack([p1 p2; p3 p4])
draw(PNG("./Plots/Model0.png"), h0)


# ----------------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
econsim0 = ModelSim(params, polfun, settings, hf, nsim = 1000000);
data0 = myunique(econsim0.sim)
DDsimulated = fill(NaN, params.ne * params.nx, 3)
DDsimulated[:, 1:2] =
    [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1))]
for i = 1:size(data0, 1)
    posb = findfirst(x -> x == data0[i, 2], settings.b)
    posy = findfirst(x -> x == data0[i, 8], settings.y)
    DDsimulated[(posy-1)*params.ne+posb, 3] = data0[i, 5]
end
heads = [:debt, :output, :D]
DDsimulated = DataFrame(Tables.table(DDsimulated, header = heads))
p4 = Gadfly.plot(
    ModelData,
    x = "debt",
    y = "output",
    color = "D",
    Geom.rectbin,
    Scale.color_discrete_manual("yellow", "black"),
    Theme(background_color = "white", key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"),
    Guide.xlabel("Debt (t)"),
    Guide.colorkey(title = "Default choice", labels = ["Default", "No Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]),
    Guide.yticks(ticks = yticks),
    Guide.title("(a) Whole grid"),
);
set_default_plot_size(18cm, 12cm)

p5 = Gadfly.plot(
    DDsimulated,
    x = "debt",
    y = "output",
    color = "D",
    Geom.rectbin,
    Scale.color_discrete_manual("white", "black", "yellow"),
    Theme(background_color = "white"),
    Theme(background_color = "white", key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"),
    Guide.xlabel("Debt (t)"),
    Guide.colorkey(
        title = "Default choice",
        labels = ["Non observed", "No Default", "Default"],
    ),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]),
    Guide.yticks(ticks = yticks),
    Guide.title("(b) Simulated Data"),
);
pdef = round(100 * sum(econsim0.sim[:, 5]) / 1000000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");



#= It gives us the first problems:
    □ The number of unique observations are small
    □ Some yellow whenm they shoul dbe black
 =#
set_default_plot_size(12cm, 12cm)
heat1 = Gadfly.vstack(p4, p5)
draw(PNG("./Plots/heat1.png"), heat1)

# **********************************************************
# [Note] To be sure that the updating code is well,
#       I input the actual value functions and verify the
#       deviations in policy functions
# **********************************************************
hat_vr = polfun.vr
hat_vd = polfun.vd
trial1 = update_solve(hat_vr, hat_vd, settings, params, uf)
difPolFun = max(maximum(abs.(trial1.bb - polfun.bb)), maximum(abs.(trial1.D - polfun.D)))
display("After updating the difference in Policy functions is : $difPolFun")

# ##########################################################
# [4] FULL INFORMATION
# ##########################################################
cheby(x, d) = begin
    mat1 = Array{Float64,2}(undef, size(x, 1), d + 1)
    mat1[:, 1:2] = [ones(size(x, 1)) x]
    for i = 3:d+1
        mat1[:, i] = 2 .* x .* mat1[:, i-1] - mat1[:, i-1]
    end
    return mat1
end

myexpansion(x1, x2,d) = begin
    matv1 = x1 .^ convert(Array, 0:d)'
    matv2 = x2 .^ convert(Array, 0:d)'
    xbasis = Array{Float64,2}(undef, size(matv1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
    startcol = 1
    for i = 0:d
        cols = d - i + 1
        endcol = startcol + cols - 1
        mati = matv1[:, i+1] .* matv2[:, 1:cols]
        xbasis[:, startcol:endcol] = mati
        startcol = endcol + 1
    end
    return xbasis
end


myexpansion(vars::Tuple,d) = begin
    nvar = size(vars,1)
    for i in 1:nvar
        aux1 = aux0
    matv2 = x2 .^ convert(Array, 0:d)'
    xbasis = Array{Float64,2}(undef, size(matv1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
    startcol = 1
    for i = 0:d
        cols = d - i + 1
        endcol = startcol + cols - 1
        mati = matv1[:, i+1] .* matv2[:, 1:cols]
        xbasis[:, startcol:endcol] = mati
        startcol = endcol + 1
    end
    return xbasis
end
hcat([a[:,i] .* b for i in 1:3]...)


# ***************************************
# [4.a] Value of repayment
# ***************************************

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Approximating using a OLS approach
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
ss = [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1))]
vr = vec(polfun.vr)
xss = [ones(params.nx * params.ne, 1) ss ss .^ 2 ss[:, 1] .* ss[:, 2]]  # bₜ, yₜ
β = (xss' * xss) \ (xss' * vr)
hatvrols = xss * β
res1 = vr - hatvrols
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Normal Basis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
sst =
    2 * (ss .- minimum(ss, dims = 1)) ./ (maximum(ss, dims = 1) - minimum(ss, dims = 1)) .-
    1
vrt = 2 * (vr .- minimum(vr)) ./ (maximum(vr) - minimum(vr)) .- 1
d = 4

matv1 = sst[:, 1] .^ convert(Array, 0:d)'
matv2 = sst[:, 2] .^ convert(Array, 0:d)'
xbasis = Array{Float64,2}(undef, size(matv1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
global startcol = 1
for i = 0:d
    cols = d - i + 1
    endcol = startcol + cols - 1
    mati = matv1[:, i+1] .* matv2[:, 1:cols]
    xbasis[:, startcol:endcol] = mati
    global startcol = endcol + 1
end
βbasis = (xbasis' * xbasis) \ (xbasis' * vrt)
hatvrbasis =
    ((1 / 2 * ((xbasis * βbasis) .+ 1)) * (maximum(vr) - minimum(vr)) .+ minimum(vr))
res2 = vr - hatvrbasis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Using Chebyshev Polynomials
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
mat1 = cheby(sst[:, 1], d)
mat2 = cheby(sst[:, 2], d)
xcheby = Array{Float64,2}(undef, size(mat1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
global startcol = 1
for i = 0:d
    cols = d - i + 1
    endcol = startcol + cols - 1
    mati = mat1[:, i+1] .* mat2[:, 1:cols]
    xcheby[:, startcol:endcol] = mati
    global startcol = endcol + 1
end
βcheby = (xcheby' * xcheby) \ (xcheby' * vrt)
hatvrcheby =
    ((1 / 2 * ((xcheby * βcheby) .+ 1)) * (maximum(vr) - minimum(vr)) .+ minimum(vr))
res3 = vr - hatvrcheby

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Neural Networks
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
d = 4
sst =
    2 * (ss .- minimum(ss, dims = 1)) ./ (maximum(ss, dims = 1) - minimum(ss, dims = 1)) .-
    1
vrt = 2 * (vr .- minimum(vr)) ./ (maximum(vr) - minimum(vr)) .- 1

dataux = repeat([sst vrt], 10, 1)
dataux = dataux[rand(1:size(dataux, 1), size(dataux, 1)), :]
traindata = Flux.Data.DataLoader((dataux[:, 1:2]', dataux[:, 3]'));


NNR1 = Chain(Dense(2, d, softplus), Dense(d, 1));
NNR2 = Chain(Dense(2, d, tanh), Dense(d, 1));
NNR3 = Chain(Dense(2, d, elu), Dense(d, 1));
NNR4 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
NNR5 = Chain(Dense(2, d, swish), Dense(d, 1));

NeuralEsti(NN, data, x, y) = begin
    mytrain(NN, data)
    hatvrNN = ((1 / 2 * (NN(x')' .+ 1)) * (maximum(y) - minimum(y)) .+ minimum(y))
    resNN = y - hatvrNN
    return hatvrNN, resNN
end

hatvrNNR1, resNN1 = NeuralEsti(NNR1, traindata, sst, vr)
hatvrNNR2, resNN2 = NeuralEsti(NNR2, traindata, sst, vr)
hatvrNNR3, resNN3 = NeuralEsti(NNR3, traindata, sst, vr)
hatvrNNR4, resNN4 = NeuralEsti(NNR4, traindata, sst, vr)
hatvrNNR5, resNN5 = NeuralEsti(NNR5, traindata, sst, vr)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Summarizing
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
println("Model 1 simple OLS: MSE $(sqrt(mean(res1.^2))) and MAE $(maximum(abs.(res1)))")
println("Model 2 OLS with normalization: MSE $(sqrt(mean(res2.^2))) and MAE $(maximum(abs.(res2)))")
println("Model 3 Chebyshev: MSE $(sqrt(mean(res3.^2))) and MAE $(maximum(abs.(res3)))")
println("Neural network 1 Softplus: MSE $(sqrt(mean(resNN1.^2))) and MAE $(maximum(abs.(resNN1)))")
println("Neural network 2 Tanh: MSE $(sqrt(mean(resNN2.^2))) and MAE $(maximum(abs.(resNN2)))")
println("Neural network 3 Elu: MSE $(sqrt(mean(resNN3.^2))) and MAE $(maximum(abs.(resNN3)))")
println("Neural network 4 Sigmoid: MSE $(sqrt(mean(resNN4.^2))) and MAE $(maximum(abs.(resNN4)))")
println("Neural network 5 Swish: MSE $(sqrt(mean(resNN5.^2))) and MAE $(maximum(abs.(resNN5)))")
sumRESID = [
    sqrt(mean(res1 .^ 2)) maximum(abs.(res1)) sqrt(mean((res1 ./ vr) .^ 2))*100 maximum(abs.(res1 ./ vr))*100
    sqrt(mean(res2 .^ 2)) maximum(abs.(res2)) sqrt(mean((res2 ./ vr) .^ 2))*100 maximum(abs.(res2 ./ vr))*100
    sqrt(mean(res3 .^ 2)) maximum(abs.(res3)) sqrt(mean((res3 ./ vr) .^ 2))*100 maximum(abs.(res3 ./ vr))*100
    sqrt(mean(resNN1 .^ 2)) maximum(abs.(resNN1)) sqrt(mean((resNN1 ./ vr) .^ 2))*100 maximum(abs.(resNN1 ./ vr))*100
    sqrt(mean(resNN2 .^ 2)) maximum(abs.(resNN2)) sqrt(mean((resNN2 ./ vr) .^ 2))*100 maximum(abs.(resNN2 ./ vr))*100
    sqrt(mean(resNN3 .^ 2)) maximum(abs.(resNN3)) sqrt(mean((resNN3 ./ vr) .^ 2))*100 maximum(abs.(resNN3 ./ vr))*100
    sqrt(mean(resNN4 .^ 2)) maximum(abs.(resNN4)) sqrt(mean((resNN4 ./ vr) .^ 2))*100 maximum(abs.(resNN4 ./ vr))*100
    sqrt(mean(resNN5 .^ 2)) maximum(abs.(resNN5)) sqrt(mean((resNN5 ./ vr) .^ 2))*100 maximum(abs.(resNN5 ./ vr))*100
]

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Plotting approximations
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
heads = [
    :debt,
    :output,
    :VRMod1,
    :ResMod1,
    :VRMod2,
    :ResMod2,
    :VRMod3,
    :ResMod3,
    :VRMod4,
    :ResMod4,
    :VRMod5,
    :ResMod5,
    :VRMod6,
    :ResMod6,
    :VRMod7,
    :ResMod7,
    :VRMod8,
    :ResMod8,
]
modls = DataFrame(Tables.table(
    [ss hatvrols res1 hatvrbasis res2 hatvrcheby res3 hatvrNNR1 resNN1 hatvrNNR2 resNN2 hatvrNNR3 resNN3 hatvrNNR4 resNN4 hatvrNNR5 resNN5],
    header = heads,
))

plotres = Array{Any,1}(undef, 8)
plotfit = Array{Any,1}(undef, 8)
for i = 1:8
    if i == 1
        plotres[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[2+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white"),
            Guide.ylabel("Model 1"),
            Guide.title("Residuals model " * string(i)),
        )
        plotfit[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[1+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Fit model " * string(i)),
        )
    else
        plotres[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[2+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Residuals model " * string(i)),
        )
        plotfit[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[1+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Fit model " * string(i)),
        )
    end
end
set_default_plot_size(24cm, 18cm)
plotres1 = gridstack([
    p1 plotres[1] plotres[2]
    plotres[3] plotres[4] plotres[5]
    plotres[6] plotres[7] plotres[8]
])
draw(PNG("./Plots/res1.png"), plotres1)
plotfit1 = gridstack([
    p1 plotfit[1] plotfit[2]
    plotfit[3] plotfit[4] plotfit[5]
    plotfit[6] plotfit[7] plotfit[8]
])
draw(PNG("./Plots/fit1.png"), plotfit1)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Policy function conditional on fit
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
set_default_plot_size(24cm, 18cm)
polfunfit = Array{Any,1}(undef, 8)
simfit = Array{Any,1}(undef, 8)
difB = Array{Float64,2}(undef, params.ne * params.nx, 8)
PFB = Array{Float64,2}(undef, params.ne * params.nx, 8)
plotdifB = Array{Any,1}(undef, 8)
plotPFB = Array{Any,1}(undef, 8)
hat_vd = polfun.vd
nrep = 100000
for i = 1:8
    hat_vrfit = reshape(modls[:, 1+2*i], params.ne, params.nx)
    polfunfit[i] = update_solve(hat_vrfit, hat_vd, settings, params, uf)
    simfit[i] = ModelSim(params, polfunfit[i], settings, hf, nsim = nrep)
    global pdef = round(100 * sum(simfit[i].sim[:, 5]) / nrep; digits = 2)
    Derror = sum(abs.(polfunfit[i].D - polfun.D)) / (params.nx * params.ne)
    PFB[:, i] = vec(polfunfit[i].bb)
    difB[:, i] = vec(polfunfit[i].bb - polfun.bb)
    display("The model $i has $pdef percent of default and a default error choice of $Derror")
end
headsB =
    [:debt, :output, :Model1, :Model2, :Model3, :Model4, :Model5, :Model6, :Model7, :Model8]
DebtPoldif = DataFrame(Tables.table([ss difB], header = headsB))
DebtPol = DataFrame(Tables.table([ss PFB], header = headsB))

for i = 1:8
    plotdifB[i] = Gadfly.plot(
        DebtPoldif,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Error in PF model " * string(i)),
    )
    plotPFB[i] = Gadfly.plot(
        DebtPol,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Debt PF model " * string(i)),
    )
end

PFBerror = gridstack([
    p3 plotdifB[1] plotdifB[2]
    plotdifB[3] plotdifB[4] plotdifB[5]
    plotdifB[6] plotdifB[7] plotdifB[8]
])   #
draw(PNG("./Plots/PFBerror.png"), PFBerror)
plotPFB = gridstack([
    p3 plotPFB[1] plotPFB[2]
    plotPFB[3] plotPFB[4] plotPFB[5]
    plotPFB[6] plotPFB[7] plotPFB[8]
])
draw(PNG("./Plots/PFB.png"), plotPFB)

################################################################
# Update with simulated data
econsim = ModelSim(params, polfun, settings, hf, nsim = 100000);
sssim = econsim.sim[:,2:3]
vrsim = econsim.sim[:,9]
xsssim = [ones(size(sssim,1),1) sssim sssim .^ 2 sssim[:, 1] .* sssim[:, 2]]# bₜ, yₜ
βsim = (xsssim' * xsssim) \ (xsssim' * vrsim)
hatvrolssim = xsssim * βsim
res1sim = vrsim - hatvrolssim

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Normal Basis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
d = 4
sstminsim = minimum(sssim, dims = 1)
sstmaxsim = maximum(sssim, dims = 1)
vrtminsim = minimum(vrsim, dims = 1)
vrtmaxsim = maximum(vrsim, dims = 1)

sstsim = 2 * (ss .- sstminsim) ./ (sstmaxsim - sstminsim) .- 1
vrtsim = 2 * (vr .- vrtminsim) ./ (vrtmaxsim - vrtminsim) .- 1

matv1 = sstsim[:, 1] .^ convert(Array, 0:d)'
matv2 = sstsim[:, 2] .^ convert(Array, 0:d)'
xbasis = Array{Float64,2}(undef, size(matv1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
global startcol = 1
for i = 0:d
    cols = d - i + 1
    endcol = startcol + cols - 1
    mati = matv1[:, i+1] .* matv2[:, 1:cols]
    xbasis[:, startcol:endcol] = mati
    global startcol = endcol + 1
end
βbasis = (xbasis' * xbasis) \ (xbasis' * vrt)
hatvrbasis =
    ((1 / 2 * ((xbasis * βbasis) .+ 1)) * (maximum(vr) - minimum(vr)) .+ minimum(vr))
res2 = vr - hatvrbasis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Using Chebyshev Polynomials
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
cheby(x, d) = begin
    mat1 = Array{Float64,2}(undef, size(x, 1), d + 1)
    mat1[:, 1:2] = [ones(size(x, 1)) x]
    for i = 3:d+1
        mat1[:, i] = 2 .* x .* mat1[:, i-1] - mat1[:, i-1]
    end
    return mat1
end
mat1 = cheby(sst[:, 1], d)
mat2 = cheby(sst[:, 2], d)
xcheby = Array{Float64,2}(undef, size(mat1, 1), div((d + 2) * (d + 1), 2)) # remember that it start at 0
global startcol = 1
for i = 0:d
    cols = d - i + 1
    endcol = startcol + cols - 1
    mati = mat1[:, i+1] .* mat2[:, 1:cols]
    xcheby[:, startcol:endcol] = mati
    global startcol = endcol + 1
end
βcheby = (xcheby' * xcheby) \ (xcheby' * vrt)
hatvrcheby =
    ((1 / 2 * ((xcheby * βcheby) .+ 1)) * (maximum(vr) - minimum(vr)) .+ minimum(vr))
res3 = vr - hatvrcheby

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Neural Networks
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
d = 4
sst =
    2 * (ss .- minimum(ss, dims = 1)) ./ (maximum(ss, dims = 1) - minimum(ss, dims = 1)) .-
    1
vrt = 2 * (vr .- minimum(vr)) ./ (maximum(vr) - minimum(vr)) .- 1

dataux = repeat([sst vrt], 10, 1)
dataux = dataux[rand(1:size(dataux, 1), size(dataux, 1)), :]
traindata = Flux.Data.DataLoader((dataux[:, 1:2]', dataux[:, 3]'));


NNR1 = Chain(Dense(2, d, softplus), Dense(d, 1));
NNR2 = Chain(Dense(2, d, tanh), Dense(d, 1));
NNR3 = Chain(Dense(2, d, elu), Dense(d, 1));
NNR4 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
NNR5 = Chain(Dense(2, d, swish), Dense(d, 1));

NeuralEsti(NN, data, x, y) = begin
    mytrain(NN, data)
    hatvrNN = ((1 / 2 * (NN(x')' .+ 1)) * (maximum(y) - minimum(y)) .+ minimum(y))
    resNN = y - hatvrNN
    return hatvrNN, resNN
end

hatvrNNR1, resNN1 = NeuralEsti(NNR1, traindata, sst, vr)
hatvrNNR2, resNN2 = NeuralEsti(NNR2, traindata, sst, vr)
hatvrNNR3, resNN3 = NeuralEsti(NNR3, traindata, sst, vr)
hatvrNNR4, resNN4 = NeuralEsti(NNR4, traindata, sst, vr)
hatvrNNR5, resNN5 = NeuralEsti(NNR5, traindata, sst, vr)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Summarizing
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
println("Model 1 simple OLS: MSE $(sqrt(mean(res1.^2))) and MAE $(maximum(abs.(res1)))")
println("Model 2 OLS with normalization: MSE $(sqrt(mean(res2.^2))) and MAE $(maximum(abs.(res2)))")
println("Model 3 Chebyshev: MSE $(sqrt(mean(res3.^2))) and MAE $(maximum(abs.(res3)))")
println("Neural network 1 Softplus: MSE $(sqrt(mean(resNN1.^2))) and MAE $(maximum(abs.(resNN1)))")
println("Neural network 2 Tanh: MSE $(sqrt(mean(resNN2.^2))) and MAE $(maximum(abs.(resNN2)))")
println("Neural network 3 Elu: MSE $(sqrt(mean(resNN3.^2))) and MAE $(maximum(abs.(resNN3)))")
println("Neural network 4 Sigmoid: MSE $(sqrt(mean(resNN4.^2))) and MAE $(maximum(abs.(resNN4)))")
println("Neural network 5 Swish: MSE $(sqrt(mean(resNN5.^2))) and MAE $(maximum(abs.(resNN5)))")
sumRESID = [
    sqrt(mean(res1 .^ 2)) maximum(abs.(res1)) sqrt(mean((res1 ./ vr) .^ 2))*100 maximum(abs.(res1 ./ vr))*100
    sqrt(mean(res2 .^ 2)) maximum(abs.(res2)) sqrt(mean((res2 ./ vr) .^ 2))*100 maximum(abs.(res2 ./ vr))*100
    sqrt(mean(res3 .^ 2)) maximum(abs.(res3)) sqrt(mean((res3 ./ vr) .^ 2))*100 maximum(abs.(res3 ./ vr))*100
    sqrt(mean(resNN1 .^ 2)) maximum(abs.(resNN1)) sqrt(mean((resNN1 ./ vr) .^ 2))*100 maximum(abs.(resNN1 ./ vr))*100
    sqrt(mean(resNN2 .^ 2)) maximum(abs.(resNN2)) sqrt(mean((resNN2 ./ vr) .^ 2))*100 maximum(abs.(resNN2 ./ vr))*100
    sqrt(mean(resNN3 .^ 2)) maximum(abs.(resNN3)) sqrt(mean((resNN3 ./ vr) .^ 2))*100 maximum(abs.(resNN3 ./ vr))*100
    sqrt(mean(resNN4 .^ 2)) maximum(abs.(resNN4)) sqrt(mean((resNN4 ./ vr) .^ 2))*100 maximum(abs.(resNN4 ./ vr))*100
    sqrt(mean(resNN5 .^ 2)) maximum(abs.(resNN5)) sqrt(mean((resNN5 ./ vr) .^ 2))*100 maximum(abs.(resNN5 ./ vr))*100
]

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Plotting approximations
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
heads = [
    :debt,
    :output,
    :VRMod1,
    :ResMod1,
    :VRMod2,
    :ResMod2,
    :VRMod3,
    :ResMod3,
    :VRMod4,
    :ResMod4,
    :VRMod5,
    :ResMod5,
    :VRMod6,
    :ResMod6,
    :VRMod7,
    :ResMod7,
    :VRMod8,
    :ResMod8,
]
modls = DataFrame(Tables.table(
    [ss hatvrols res1 hatvrbasis res2 hatvrcheby res3 hatvrNNR1 resNN1 hatvrNNR2 resNN2 hatvrNNR3 resNN3 hatvrNNR4 resNN4 hatvrNNR5 resNN5],
    header = heads,
))

plotres = Array{Any,1}(undef, 8)
plotfit = Array{Any,1}(undef, 8)
for i = 1:8
    if i == 1
        plotres[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[2+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white"),
            Guide.ylabel("Model 1"),
            Guide.title("Residuals model " * string(i)),
        )
        plotfit[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[1+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Fit model " * string(i)),
        )
    else
        plotres[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[2+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Residuals model " * string(i)),
        )
        plotfit[i] = Gadfly.plot(
            modls,
            x = "debt",
            y = heads[1+2*i],
            color = "output",
            Geom.line,
            Theme(background_color = "white", key_position = :none),
            Guide.ylabel("Model " * string(i)),
            Guide.title("Fit model " * string(i)),
        )
    end
end
set_default_plot_size(24cm, 18cm)
plotres1 = gridstack([
    p1 plotres[1] plotres[2]
    plotres[3] plotres[4] plotres[5]
    plotres[6] plotres[7] plotres[8]
])
draw(PNG("./Plots/res1.png"), plotres1)
plotfit1 = gridstack([
    p1 plotfit[1] plotfit[2]
    plotfit[3] plotfit[4] plotfit[5]
    plotfit[6] plotfit[7] plotfit[8]
])
draw(PNG("./Plots/fit1.png"), plotfit1)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Policy function conditional on fit
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
set_default_plot_size(24cm, 18cm)
polfunfit = Array{Any,1}(undef, 8)
simfit = Array{Any,1}(undef, 8)
difB = Array{Float64,2}(undef, params.ne * params.nx, 8)
PFB = Array{Float64,2}(undef, params.ne * params.nx, 8)
plotdifB = Array{Any,1}(undef, 8)
plotPFB = Array{Any,1}(undef, 8)
hat_vd = polfun.vd
nrep = 100000
for i = 1:8
    hat_vrfit = reshape(modls[:, 1+2*i], params.ne, params.nx)
    polfunfit[i] = update_solve(hat_vrfit, hat_vd, settings, params, uf)
    simfit[i] = ModelSim(params, polfunfit[i], settings, hf, nsim = nrep)
    global pdef = round(100 * sum(simfit[i].sim[:, 5]) / nrep; digits = 2)
    Derror = sum(abs.(polfunfit[i].D - polfun.D)) / (params.nx * params.ne)
    PFB[:, i] = vec(polfunfit[i].bb)
    difB[:, i] = vec(polfunfit[i].bb - polfun.bb)
    display("The model $i has $pdef percent of default and a default error choice of $Derror")
end
headsB =
    [:debt, :output, :Model1, :Model2, :Model3, :Model4, :Model5, :Model6, :Model7, :Model8]
DebtPoldif = DataFrame(Tables.table([ss difB], header = headsB))
DebtPol = DataFrame(Tables.table([ss PFB], header = headsB))

for i = 1:8
    plotdifB[i] = Gadfly.plot(
        DebtPoldif,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Error in PF model " * string(i)),
    )
    plotPFB[i] = Gadfly.plot(
        DebtPol,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Debt PF model " * string(i)),
    )
end

PFBerror = gridstack([
    p3 plotdifB[1] plotdifB[2]
    plotdifB[3] plotdifB[4] plotdifB[5]
    plotdifB[6] plotdifB[7] plotdifB[8]
])   #
draw(PNG("./Plots/PFBerror.png"), PFBerror)
plotPFB = gridstack([
    p3 plotPFB[1] plotPFB[2]
    plotPFB[3] plotPFB[4] plotPFB[5]
    plotPFB[6] plotPFB[7] plotPFB[8]
])
draw(PNG("./Plots/PFB.png"), plotPFB)

################################################################
