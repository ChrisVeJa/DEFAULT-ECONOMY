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
    Gadfly
using Cairo, Fontconfig, Tables, DataFrames, Compose
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
mytrain(NN, data) = begin
    lossf(x, y) = Flux.mse(NN(x), y)
    pstrain = Flux.params(NN)
    Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, data, Descent())
end
# ----------------------------------------------------------
# [1.d]  Policy function conditional on expected values
# ----------------------------------------------------------
update_solve(hat_vr, hat_vd, settings, params, uf) = begin
    # Starting the psolution of the model
    @unpack P, b, y = settings
    @unpack r, β, ne, nx, σrisk, θ = params
    p0 = findmin(abs.(0 .- b))[2]
    udef = repeat(settings.udef', ne, 1)
    hat_vf = max.(hat_vr, hat_vd)
    hat_D = 1 * (hat_vd .> hat_vr)
    evf1 = hat_vf * P'
    evd1 = hat_vd * P'
    eδD1 = hat_D * P'
    q1 = (1 / (1 + r)) * (1 .- eδD1) # price
    qb1 = q1 .* b
    βevf1 = β * evf1
    vrnew = Array{Float64,2}(undef, ne, nx)
    cc1 = Array{Float64,2}(undef, ne, nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    yb = b .+ y'
    @inbounds for i = 1:ne
        cc1 = yb[i, :]' .- qb1
        cc1 = max.(cc1, 0)
        aux_u = uf.(cc1, σrisk) + βevf1
        vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
    end
    bb = repeat(b, 1, nx)
    bb = bb[bpnew]
    evaux = θ * evf1[p0, :]' .+ (1 - θ) * evd1
    vdnew = udef + β * evaux
    vfnew = max.(vrnew, vdnew)
    Dnew = 1 * (vdnew .> vrnew)
    eδD = Dnew * P'
    qnew = (1 / (1 + r)) * (1 .- eδD)
    return (vf = vfnew, vr = vrnew, vd = vdnew, D = Dnew, bb = bb, q = qnew, bp = bpnew)
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
)
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
plots0 = Array{Any,1}(undef, 6)
vars = ["vf" "vr" "vd" "b"]
titlevars =
    ["Value function" "Value of rapayment" "Value of dfefault" "Policy function for debt"]
for i = 1:length(vars)
    plots0[i] = Gadfly.plot(
        ModelData,
        x = "debt",
        y = vars[i],
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
        Guide.title(titlevars[i]),
    )
end
Gadfly.draw(PNG("./Plots/ValuFunction.png"), plots0[1]);


set_default_plot_size(12cm, 8cm)
ytick = round.(settings.y, digits = 2)
yticks = [ytick[1], ytick[6], ytick[11], ytick[16], ytick[end]]
plots0[5] = Gadfly.plot(
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
    Guide.title("Default Choice"),
);
set_default_plot_size(18cm, 12cm)
h0 = Gadfly.gridstack([plots0[2] plots0[3]; plots0[4] plots0[5]])
Gadfly.draw(PNG("./Plots/Model0.png"), h0)

# ----------------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
Nsim = 1000000
econsim0 = ModelSim(params, polfun, settings, hf, nsim = Nsim, burn = 0, ini_st = [1 0 1]);
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
sort!(DDsimulated, :D)
plots0[6] = Gadfly.plot(
    DDsimulated,
    x = "debt",
    y = "output",
    color = "D",
    Geom.rectbin,
    Scale.color_discrete_manual("black", "yellow", "white"),
    Theme(background_color = "white"),
    Theme(background_color = "white", key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"),
    Guide.xlabel("Debt (t)"),
    Guide.colorkey(
        title = "Default choice",
        labels = ["No Default", "Default", "Non observed"],
    ),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]),
    Guide.yticks(ticks = yticks),
    Guide.title("Default choice: Simulated Data"),
);
pdef = round(100 * sum(econsim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

#= It gives us the first problems:
    □ The number of unique observations are small
    □ Some yellow whenm they shoul dbe black
 =#
set_default_plot_size(12cm, 12cm)
heat1 = Gadfly.vstack(plots0[5], plots0[6])
Gadfly.draw(PNG("./Plots/heat1.png"), heat1)

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
result = Array{Any,2}(undef, 8, 2) # [fit, residual]


# ##########################################################################################

#                 [4] ESTIMATING VALUE OF REPAYMENT WITH FULL GRID

# ##########################################################################################

cheby(x, d) = begin
    mat1 = Array{Float64,2}(undef, size(x, 1), d + 1)
    mat1[:, 1:2] = [ones(size(x, 1)) x]
    for i = 3:d+1
        mat1[:, i] = 2 .* x .* mat1[:, i-1] - mat1[:, i-1]
    end
    return mat1
end

myexpansion(vars::Tuple, d) = begin
    nvar = size(vars, 1)
    auxi = vars[1]
    numi = convert(Array, 0:size(auxi, 2)-1)'
    for i = 2:nvar
        n2v = size(auxi, 2)
        auxi2 = hcat([auxi[:, j] .* vars[i] for j = 1:n2v]...)
        numi2 = hcat([
            numi[:, j] .+ convert(Array, 0:size(vars[i], 2)-1)' for j = 1:n2v
        ]...)
        auxi = auxi2
        numi = numi2
    end
    xbasis = vcat(numi, auxi)
    xbasis = xbasis[:, xbasis[1, :].<=d]
    xbasis = xbasis[2:end, :]
    return xbasis
end

NeuralEsti(NN, data, x, y) = begin
    mytrain(NN, data)
    hatvrNN = ((1 / 2 * (NN(x')' .+ 1)) * (maximum(y) - minimum(y)) .+ minimum(y))
    resNN = y - hatvrNN
    return hatvrNN, resNN
end

# ***************************************
# [.a]  PRELIMINARIES
# ***************************************
ss = [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1))]
vr = vec(polfun.vr)
ssmin = minimum(ss, dims = 1)
ssmax = maximum(ss, dims = 1)
vrmin = minimum(vr)
vrmax = maximum(vr)
sst = 2 * (ss .- ssmin) ./ (ssmax - ssmin) .- 1
vrt = 2 * (vr .- vrmin) ./ (vrmax - vrmin) .- 1
d = 4
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# [.b] a OLS approach
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
xs1 = [ones(params.nx * params.ne, 1) ss ss .^ 2 ss[:, 1] .* ss[:, 2]]  # bₜ, yₜ
β1 = (xs1' * xs1) \ (xs1' * vr)
result[1, 1] = xs1 * β1
result[1, 2] = vr - result[1, 1]
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Power Basis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
mat = (sst[:, 1] .^ convert(Array, 0:d)', sst[:, 2] .^ convert(Array, 0:d)')
xs2 = myexpansion(mat, d)
β2 = (xs2' * xs2) \ (xs2' * vrt)
result[2, 1] = ((1 / 2 * ((xs2 * β2) .+ 1)) * (vrmax - vrmin) .+ vrmin)
result[2, 2] = vr - result[2, 1]
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Chebyshev Polynomials
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
mat = (cheby(sst[:, 1], d), cheby(sst[:, 2], d))
xs3 = myexpansion(mat, d) # remember that it start at 0
β3 = (xs3' * xs3) \ (xs3' * vrt)
result[3, 1] = ((1 / 2 * ((xs3 * β3) .+ 1)) * (vrmax - vrmin) .+ vrmin)
result[3, 2] = vr - result[3, 1]

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Neural Networks
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
dataux = repeat([sst vrt], 10, 1)
dataux = dataux[rand(1:size(dataux, 1), size(dataux, 1)), :]
traindata = Flux.Data.DataLoader((dataux[:, 1:2]', dataux[:, 3]'));

NNR1 = Chain(Dense(2, d, softplus), Dense(d, 1));
NNR2 = Chain(Dense(2, d, tanh), Dense(d, 1));
NNR3 = Chain(Dense(2, d, elu), Dense(d, 1));
NNR4 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
NNR5 = Chain(Dense(2, d, swish), Dense(d, 1));

result[4, 1], result[4, 2] = NeuralEsti(NNR1, traindata, sst, vr)
result[5, 1], result[5, 2] = NeuralEsti(NNR2, traindata, sst, vr)
result[6, 1], result[6, 2] = NeuralEsti(NNR3, traindata, sst, vr)
result[7, 1], result[7, 2] = NeuralEsti(NNR4, traindata, sst, vr)
result[8, 1], result[8, 2] = NeuralEsti(NNR5, traindata, sst, vr)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Summarizing results
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
sumR = Array{Float32,2}(undef, 8, 4)
f1(x) = sqrt(mean(x .^ 2))                # Square root of Mean Square Error
f2(x) = maximum(abs.(x))                  # Maximum Absolute Deviation
f3(x, y) = sqrt(mean((x ./ y) .^ 2)) * 100     # Square root of Mean Relative Square Error
f4(x, y) = maximum(abs.(x ./ y)) * 100         # Maximum Relative deviation
for i = 1:size(sumR, 1)
    sumR[i, :] =
        [f1(result[i, 2]) f2(result[i, 2]) f3(result[i, 2], vr) f4(result[i, 2], vr)]
end

#   Then the position ij of the matrix sumR is the f_j(residual_mdoel_i)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Plotting approximations
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
heads = [
    :debt,
    :output,
    :VR1,
    :VR2,
    :VR3,
    :VR4,
    :VR5,
    :VR6,
    :VR7,
    :VR8,
    :Res1,
    :Res2,
    :Res3,
    :Res4,
    :Res5,
    :Res6,
    :Res7,
    :Res8,
]
modls = DataFrame(Tables.table([ss hcat(result...)], header = heads))
models = ["OLS" "Power series" "Chebyshev" "Softplus" "Tanh" "Elu" "Sigmoid" "Swish"]
plots1 = Array{Any,2}(undef, 8, 2) # [̂vr ϵ] by model
for i = 1:8
    plots1[i, 1] = Gadfly.plot(
        modls,
        x = "debt",
        y = heads[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel(""),
        Guide.title(models[i]),
    )
    plots1[i, 2] = Gadfly.plot(
        modls,
        x = "debt",
        y = heads[10+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel(""),
        Guide.title(models[i]),
    )
end
set_default_plot_size(24cm, 18cm)
plotfit1 = gridstack([
    plots0[2] plots1[1, 1] plots1[2, 1]
    plots1[3, 1] plots1[4, 1] plots1[5, 1]
    plots1[6, 1] plots1[7, 1] plots1[8, 1]
])
plotres1 = gridstack([
    plots0[2] plots1[1, 2] plots1[2, 2]
    plots1[3, 2] plots1[4, 2] plots1[5, 2]
    plots1[6, 2] plots1[7, 2] plots1[8, 2]
])
draw(PNG("./Plots/res1.png"), plotres1)
draw(PNG("./Plots/fit1.png"), plotfit1)
# plotfit1 :=  Plots for the estimated  value of repayment
# plotres1 :=  Residuals

# ==============================================================================
#          UPDATING POLICY FUNCTIONS BASED ON PREVIOUS ESTIMATIONS
# ==============================================================================
set_default_plot_size(24cm, 18cm)
hat_vd = polfun.vd
ResultUp = Array{Float64,2}(undef, params.ne * params.nx, 8 * 2)
for i = 1:8
    hat_vrfit = reshape(modls[:, 2+i], params.ne, params.nx)
    polfunfit = update_solve(hat_vrfit, hat_vd, settings, params, uf)
    ResultUp[:, i] = vec(polfunfit.bb)
    ResultUp[:, i+8] = vec(polfunfit.bb - polfun.bb)
    # Simulation
    simfit = ModelSim(params, polfunfit, settings, hf, nsim = Nsim)
    pdef1 = round(100 * sum(simfit.sim[:, 5]) / Nsim; digits = 2)
    Derror = sum(abs.(polfunfit.D - polfun.D)) / (params.nx * params.ne)
    display("The model $i has $pdef1 percent of default and a default error choice of $Derror")
end
headsB = [
    :debt,
    :output,
    :PF1,
    :PF2,
    :PF3,
    :PF4,
    :PF5,
    :PF6,
    :PF7,
    :PF8,
    :Rs1,
    :Rs2,
    :Rs3,
    :Rs4,
    :Rs5,
    :Rs6,
    :Rs7,
    :Rs8,
]
DebtPolUp = DataFrame(Tables.table([ss ResultUp], header = headsB))
plotPolUp = Array{Any,2}(undef, 8, 2)

for i = 1:8
    plotPolUp[i, 1] = Gadfly.plot(
        DebtPolUp,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Debt PF model " * string(i)),
    )
    plotPolUp[i, 2] = Gadfly.plot(
        DebtPolUp,
        x = "debt",
        y = headsB[10+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Error in PF model " * string(i)),
    )
end
PlotPFB = gridstack([
    plots0[2] plotPolUp[1, 1] plotPolUp[2, 1]
    plotPolUp[3, 1] plotPolUp[4, 1] plotPolUp[5, 1]
    plotPolUp[6, 1] plotPolUp[7, 1] plotPolUp[8, 1]
])   #
PFBerror = gridstack([
    plots0[2] plotPolUp[1, 2] plotPolUp[2, 2]
    plotPolUp[3, 2] plotPolUp[4, 2] plotPolUp[5, 2]
    plotPolUp[6, 2] plotPolUp[7, 2] plotPolUp[8, 2]
])   #
draw(PNG("./Plots/PFB.png"), PlotPFB)
draw(PNG("./Plots/PFBerror.png"), PFBerror)

################################################################
# 5. WITH SIMULATED DATA
################################################################
econsim = ModelSim(params, polfun, settings, hf, nsim = 100000);
ss1 = econsim.sim[:, 2:3]
vr1 = econsim.sim[:, 9]
ss1min = minimum(ss1, dims = 1)
ss1max = maximum(ss1, dims = 1)
vr1min = minimum(vr1)
vr1max = maximum(vr1)
sst1 = 2 * (ss1 .- ss1min) ./ (ss1max - ss1min) .- 1
vrt1 = 2 * (vr1 .- vr1min) ./ (vr1max - vr1min) .- 1
resultS = Array{Any,2}(undef, 8, 2) # [fit, residual]
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Approximating using a OLS approach
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
xs1s = [ones(size(ss1, 1), 1) ss1 ss1 .^ 2 ss1[:, 1] .* ss1[:, 2]]  # bₜ, yₜ
β1s = (xs1s' * xs1s) \ (xs1s' * vr1)
resultS[1, 1] = xs1 * β1s     ###  Original grid
resultS[1, 2] = vr - resultS[1, 1]
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Normal Basis
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
mats = (sst1[:, 1] .^ convert(Array, 0:d)', sst1[:, 2] .^ convert(Array, 0:d)')
xs2s = myexpansion(mats, d)
β2s = (xs2s' * xs2s) \ (xs2s' * vrt1)
resultS[2, 1] = ((1 / 2 * ((xs2 * β2s) .+ 1)) * (vrmax - vrmin) .+ vrmin)
resultS[2, 2] = vr - resultS[2, 1]
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Using Chebyshev Polynomials
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
mats = (cheby(sst1[:, 1], d), cheby(sst1[:, 2], d))
xs3s = myexpansion(mats, d) # remember that it start at 0
β3s = (xs3s' * xs3s) \ (xs3s' * vrt1)
resultS[3, 1] = ((1 / 2 * ((xs3 * β3s) .+ 1)) * (vrmax - vrmin) .+ vrmin)
resultS[3, 2] = vr - resultS[3, 1]

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Neural Networks
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
traindatas = Flux.Data.DataLoader((sst1', vrt1'));

NNR1s = Chain(Dense(2, d, softplus), Dense(d, 1));
NNR2s = Chain(Dense(2, d, tanh), Dense(d, 1));
NNR3s = Chain(Dense(2, d, elu), Dense(d, 1));
NNR4s = Chain(Dense(2, d, sigmoid), Dense(d, 1));
NNR5s = Chain(Dense(2, d, swish), Dense(d, 1));

resultS[4, 1], resultS[4, 2] = NeuralEsti(NNR1s, traindatas, sst, vr)
resultS[5, 1], resultS[5, 2] = NeuralEsti(NNR2s, traindatas, sst, vr)
resultS[6, 1], resultS[6, 2] = NeuralEsti(NNR3s, traindatas, sst, vr)
resultS[7, 1], resultS[7, 2] = NeuralEsti(NNR4s, traindatas, sst, vr)
resultS[8, 1], resultS[8, 2] = NeuralEsti(NNR5s, traindatas, sst, vr)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Summarizing
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
sumRs = Array{Float32,2}(undef, 8, 4)
for i = 1:size(sumR, 1)
    sumRs[i, :] =
        [f1(resultS[i, 2]) f2(resultS[i, 2]) f3(resultS[i, 2], vr) f4(resultS[i, 2], vr)]
end

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Plotting approximations
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
heads = [
    :debt,
    :output,
    :VR1,
    :VR2,
    :VR3,
    :VR4,
    :VR5,
    :VR6,
    :VR7,
    :VR8,
    :Res1,
    :Res2,
    :Res3,
    :Res4,
    :Res5,
    :Res6,
    :Res7,
    :Res8,
]
modlsS = DataFrame(Tables.table([ss hcat(resultS...)], header = heads))
modelsS = ["OLS" "Power series" "Chebyshev" "Softplus" "Tanh" "Elu" "Sigmoid" "Swish"]
plots1S = Array{Any,2}(undef, 8, 2) # [fit, residual]
for i = 1:8
    plots1S[i, 1] = Gadfly.plot(
        modlsS,
        x = "debt",
        y = heads[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel(""),
        Guide.title(modelsS[i]),
    )
    plots1S[i, 2] = Gadfly.plot(
        modlsS,
        x = "debt",
        y = heads[10+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel(""),
        Guide.title(modelsS[i]),
    )
end
set_default_plot_size(24cm, 18cm)
plotfit1S = gridstack([
    plots0[2] plots1S[1, 1] plots1S[2, 1]
    plots1S[3, 1] plots1S[4, 1] plots1S[5, 1]
    plots1S[6, 1] plots1S[7, 1] plots1S[8, 1]
])
plotres1S = gridstack([
    plots0[2] plots1S[1, 2] plots1S[2, 2]
    plots1S[3, 2] plots1S[4, 2] plots1S[5, 2]
    plots1S[6, 2] plots1S[7, 2] plots1S[8, 2]
])
draw(PNG("./Plots/res1S.png"), plotres1S)
draw(PNG("./Plots/fit1S.png"), plotfit1S)

# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# UPDATING POLICY FUNCTIONS
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
set_default_plot_size(24cm, 18cm)
hat_vd = polfun.vd
ResultSUp = Array{Float64,2}(undef, params.ne * params.nx, 8 * 2)
for i = 1:8
    hat_vrSfit = reshape(modlsS[:, 2+i], params.ne, params.nx)
    polfunSfit = update_solve(hat_vrSfit, hat_vd, settings, params, uf)
    ResultSUp[:, i] = vec(polfunSfit.bb)
    ResultSUp[:, i+8] = vec(polfunSfit.bb - polfun.bb)
    # Simulation
    simfit = ModelSim(params, polfunSfit, settings, hf, nsim = Nsim)
    pdef1 = round(100 * sum(simfit.sim[:, 5]) / Nsim; digits = 2)
    Derror = sum(abs.(polfunSfit.D - polfun.D)) / (params.nx * params.ne)
    display("The model $i has $pdef1 percent of default and a default error choice of $Derror")
end
headsB = [
    :debt,
    :output,
    :PF1,
    :PF2,
    :PF3,
    :PF4,
    :PF5,
    :PF6,
    :PF7,
    :PF8,
    :Rs1,
    :Rs2,
    :Rs3,
    :Rs4,
    :Rs5,
    :Rs6,
    :Rs7,
    :Rs8,
]
DebtPolSUp = DataFrame(Tables.table([ss ResultSUp], header = headsB))
plotPolSUp = Array{Any,2}(undef, 8, 2)

for i = 1:8
    plotPolSUp[i, 2] = Gadfly.plot(
        DebtPolSUp,
        x = "debt",
        y = headsB[2+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Debt PF model " * string(i)),
    )
    plotPolSUp[i, 1] = Gadfly.plot(
        DebtPolSUp,
        x = "debt",
        y = headsB[10+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel("Model " * string(i)),
        Guide.title("Error in PF model " * string(i)),
    )
end
PlotSPFB = gridstack([
    plots0[4] plotPolSUp[1, 1] plotPolSUp[2, 1]
    plotPolSUp[3, 1] plotPolSUp[4, 1] plotPolSUp[5, 1]
    plotPolSUp[6, 1] plotPolSUp[7, 1] plotPolSUp[8, 1]
])   #
PFBSerror = gridstack([
    plots0[4] plotPolSUp[1, 2] plotPolSUp[2, 2]
    plotPolSUp[3, 2] plotPolSUp[4, 2] plotPolSUp[5, 2]
    plotPolSUp[6, 2] plotPolSUp[7, 2] plotPolSUp[8, 2]
])   #
draw(PNG("./Plots/PFBS.png"), PlotSPFB)
draw(PNG("./Plots/PFBSerror.png"), PFBSerror)
