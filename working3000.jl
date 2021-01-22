###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################
using Random, Distributions, Statistics, LinearAlgebra, StatsBase
using Parameters, Flux, ColorSchemes, Gadfly
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
    nx = (2*(x .- lx) ./ (ux-lx)) .-1
    #nx = (x .- 0.5 * (ux + lx)) ./ (0.5 * (ux - lx))
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
#                   [2] THE MODEL
############################################################
# ----------------------------------------------------------
# [2.a] Settings
# ----------------------------------------------------------

params = (r = 0.017, σrisk = 2.0, ρ = 0.945,η = 0.025,β = 0.953,θ = 0.282,nx = 21,
        m = 3,μ = 0.0, fhat = 0.969,ub = 0,lb = -0.4,tol = 1e-8, maxite = 500, ne = 1001)
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))
# ----------------------------------------------------------
# [2.b] Solving the model
# ----------------------------------------------------------
@time polfun, settings = Solver(params, hf, uf);
MoDel = [vec(polfun[i]) for i = 1:6]
MoDel = [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1)) hcat(MoDel...)]
heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q]
ModelData = DataFrame(Tables.table(MoDel, header = heads))
# ----------------------------------------------------------
# [2.d] Simulating data from  the model
# ----------------------------------------------------------
Nsim = 100000
econsim0 = ModelSim(params, polfun, settings, hf, nsim = Nsim);
pdef = round(100 * sum(econsim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
ytick = round.(settings.y, digits = 2)
yticks = [ytick[1], ytick[6], ytick[11], ytick[16], ytick[end]]
myheat(mysimul, myparams, mysettings ) = begin
    data0 = myunique(mysimul.sim)
    DDsimulated = fill(NaN, myparams.ne * myparams.nx, 3)
    DDsimulated[:, 1:2] = [repeat(mysettings.b, myparams.nx) repeat(mysettings.y, inner = (myparams.ne, 1))]
    for i = 1:size(data0, 1)
        posb = findfirst(x -> x == data0[i, 2], mysettings.b)
        posy = findfirst(x -> x == data0[i, 8], mysettings.y)
        DDsimulated[(posy-1)*params.ne+posb, 3] = data0[i, 5]
    end
    heads = [:debt, :output, :D]
    DDsimulated = DataFrame(Tables.table(DDsimulated, header = heads))
    sort!(DDsimulated, :D)
    myploty = Gadfly.plot(
            DDsimulated, x = "debt", y = "output", color = "D", Geom.rectbin, Scale.color_discrete_manual("green", "red", "white"),
            Theme(background_color = "white"), Theme(background_color = "white", key_title_font_size = 8pt, key_label_font_size = 8pt),
            Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
            Guide.colorkey(title = "Default choice",labels = ["No Default", "Default", "Non observed"]),
            Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]), Guide.yticks(ticks = yticks), Guide.title("Default choice: Simulated Data"),
            );
    return myploty
end
heat = myheat(econsim0, params, settings )
# ###############################################################
# [3] ESTIMATING VALUE OF REPAYMENT WITH WITH SIMULATED DATA
# ###############################################################
# Estimation
#   Simulation:
# θ = 1 means that the country is out the market
# Taking just when the country is in the market
# Simulating
Nsim= 100_000
econsim = ModelSim(params, polfun, settings, hf, nsim = Nsim);
modsim = econsim.sim[econsim.sim[:,end].==0,:]
ss1, uss1, lss1 = mynorm(modsim[:, 2:3])
vr1, uvr1, lvr1 = mynorm(modsim[:, 9])
d=16
NNR1 = Chain(Dense(2, d, softplus), Dense(d, 1));
loss(x, y) = Flux.mse(NNR1(x), y)




NeuralEsti(NN, data, x, y) = begin
    mytrain(NN, data)
    hatvrNN = ((1 / 2 * (NN(x')' .+ 1)) .* (y[1] - y[2])) .+ y[2]
    return hatvrNN
end

# All the states of the grid
ss = [repeat(settings.b,params.nx) repeat(settings.y, params.ne)]
ss = (2*(ss .- lss1) ./(uss1 - lss1)) .- 1


# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Neural Networks
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
traindatas = Flux.Data.DataLoader((ss1', vr1'));
d = 16
NNR1 = Chain(Dense(2, d, softplus), Dense(d, 1));
NNR2 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
vrhatNN1 = NeuralEsti(NNR1, traindatas, ss, [uvr1 lvr1])
vrhatNN2 = NeuralEsti(NNR2, traindatas, ss, [uvr1 lvr1])
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
# Plotting approximations
# ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
heads   = [:debt,:output,:VR1,:VR2,:VR3,:Res1,:Res2,:Res3]
modlsS  = DataFrame(Tables.table([ss hcat(resultS...)], header = heads))
modelsS = ["Chebyshev" "Softplus"  "Sigmoid"]
plots1S = Array{Any,2}(undef, 3, 2) # [fit, residual]
for i = 1:3
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
        y = heads[5+i],
        color = "output",
        Geom.line,
        Theme(background_color = "white", key_position = :none),
        Guide.ylabel(""),
        Guide.title(modelsS[i]),
    )
end
set_default_plot_size(24cm, 18cm)
plotfit1S = gridstack([plots0[2] plots1S[1, 1] ; plots1S[2, 1] plots1S[3, 1]])
plotres1S = gridstack([plots0[2] plots1S[1, 2] ; plots1S[2, 2] plots1S[3, 2]])
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
headsB = [:debt,:output,:PF1,:PF2,:PF3,:PF4,:PF5,:PF6,:PF7,:PF8,
        :Rs1,:Rs2,:Rs3,:Rs4,:Rs5,:Rs6,:Rs7,:Rs8,
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
