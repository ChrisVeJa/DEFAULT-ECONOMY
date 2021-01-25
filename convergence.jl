###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
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
f1(x) = sqrt(mean(x .^ 2))                     # Square root of Mean Square Error
f2(x) = maximum(abs.(x))                       # Maximum Absolute Deviation
f3(x, y) = sqrt(mean((x ./ y) .^ 2)) * 100     # Square root of Mean Relative Square Error
f4(x, y) = maximum(abs.(x ./ y)) * 100         # Maximum Relative deviation
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
        DDsimulated,
        x = "debt",
        y = "output",
        color = "D",
        Geom.rectbin,
        Scale.color_discrete_manual("green", "red", "white"),
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
    return myploty
end
plot2(DatFra,nmod,names, heads) = begin
    plots1S = Array{Any,2}(undef,nmod,2)
    for i = 1:nmod
        for j = 1:2
            plots1S[i, j] = Gadfly.plot(
                DatFra,
                x = "debt",
                y = heads[2+(j-1)*nmod+i],
                color = "output",
                Geom.line,
                Theme(background_color = "white", key_position = :none),
                Guide.ylabel(""),
                Guide.title(names[i]),
            )
        end
    end
    return plots1S
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
    ne = 1001,
)
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
@time polfun, settings = Solver(params, hf, uf);
Nsim = 1000000
econsim0 = ModelSim(params, polfun, settings, hf, nsim = Nsim);
pdef = round(100 * sum(econsim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
# ---------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
Nsim = 100_000
d = 4; nmod = 3; dNN = 16
ss = [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1))]
vr = vec(polfun.vr)
ssmin = minimum(ss, dims = 1);   ssmax = maximum(ss, dims = 1)
vrmin = minimum(vr) ;vrmax = maximum(vr)
sst = 2 * (ss .- ssmin) ./ (ssmax - ssmin) .- 1
vrt = 2 * (vr .- vrmin) ./ (vrmax - vrmin) .- 1
mat = (cheby(sst[:, 1], d), cheby(sst[:, 2], d))
xs = myexpansion(mat, d)

mydata(data) = begin
    ss1 = data[1];
    vr1 = data[2];
    ss1min = minimum(ss1, dims = 1);
    ss1max = maximum(ss1, dims = 1);
    vr1min = minimum(vr1);
    vr1max = maximum(vr1);
    sst1 = 2 * (ss1 .- ss1min) ./ (ss1max - ss1min) .- 1
    vrt1 = 2 * (vr1 .- vr1min) ./ (vr1max - vr1min) .- 1
    return sst1,vrt1
end

myupd(data1, vrmax, vrmin,d, Nsim, params,polfun, settings,uf,hf) = begin
     sst1,vrt1 = mydata(data1)
     mats = (cheby(sst1[:, 1], d), cheby(sst1[:, 2], d))
     xss = myexpansion(mats, d) # remember that it start at 0
     βs = (xss' * xss) \ (xss' * vrt1)
  # Projection
     vrhat1 = ((1 / 2 * ((xs * βs) .+ 1)) * (vrmax - vrmin) .+ vrmin)
     vrhat1 = reshape(vrhat1,params.ne, params.nx)
  # Updating
     polfunS = update_solve(vrhat1, polfun.vd, settings, params, uf)
  # Simulation
     simaux = ModelSim(params, polfunS, settings, hf, nsim = Nsim)
     pdef = round(100 * sum(simaux.sim[:, 5]) / Nsim; digits = 2);
     datanew = (simaux.sim[simaux.sim[:,end].== 0,2:3], simaux.sim[simaux.sim[:,end].== 0,9])
     display("Simulation finished, with frequency of $pdef default events");
     return βs,datanew
end
d = 4
dif1 = 1
data1 = nothing
β0 = zeros(15,1)
for i in 1:100
    if i ==1
        data1 = (econsim0.sim[econsim0.sim[:,end].== 0,2:3], econsim0.sim[econsim0.sim[:,end].== 0,9])
        β1,data1 = myupd(data1, vrmax, vrmin,d, Nsim, params,polfun, settings,uf,hf)
        difnew = maximum(abs.(β1- β0))
        β0 = 0.9*β1 + 0.1*β0
    else
        β1,data1 = myupd(data1, vrmax, vrmin,d, Nsim, params,polfun, settings,uf,hf)
        difnew = maximum(abs.(β1- β0))
        β0 = 0.9*β1 + 0.1*β0
    end
    display("Maximum diference in parameters: $difnew");
end
# ##################################################
    # ∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘∘
    # Neural Networks
    traindatas = Flux.Data.DataLoader((sst1', vrt1'));
    NNR1s = Chain(Dense(2, dNN, softplus), Dense(dNN, 1));
    NNR2s = Chain(Dense(2, dNN, sigmoid), Dense(dNN, 1));
    vrhat2, = NeuralEsti(NNR1s, traindatas, sst, vr)
    vrhat3, = NeuralEsti(NNR2s, traindatas, sst, vr)
