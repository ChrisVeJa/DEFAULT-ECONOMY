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
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0, fhat = 0.969, ub = 0,
        lb = -0.4, tol = 1e-8, maxite = 500, ne = 1001)
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
@time polfun, settings = Solver(params, hf, uf);
Nsim = 100_000
econsim0 = ModelSim(params, polfun, settings, hf, nsim = Nsim);
pdef = round(100 * sum(econsim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
# ---------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
d = 4; nmod = 3; dNN = 16
ss = [repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne, 1))]
vr = vec(polfun.vr)
ssmin = minimum(ss, dims = 1);   ssmax = maximum(ss, dims = 1)
vrmin = minimum(vr) ;vrmax = maximum(vr)
sst = 2 * (ss .- ssmin) ./ (ssmax - ssmin) .- 1
vrt = 2 * (vr .- vrmin) ./ (vrmax - vrmin) .- 1


x = [econsim0.sim[econsim0.sim[:,end].== 0,2:3] econsim0.sim[econsim0.sim[:,end].== 0,9]]
lx = minimum(x,dims=1)
ux = maximum(x,dims=1)
x0 = 2 * (x .- lx) ./ (ux - lx) .- 1
x0Tu = [Tuple(x0[i, :]) for i = 1:size(x0, 1)]
x1Tu = unique(x0Tu)
xx = countmap(x0Tu)
w  = [get(xx, i, 0) for i in x1Tu]
x1 = Array{Float32,2}(undef,size(x1Tu)[1],3)
for i in 1:size(x1Tu)[1]
    x1[i,:] = [x1Tu[i]...]'
end
ss1 = x1[:,1:2]'
yy  = x1[:,3]
dNN=16
NNR1s = Chain(Dense(2, dNN, softplus), Dense(dNN, 1));
loss(x,w,y) = sum(w .* ((NNR1s(x)' - y).^2))
ps = Flux.Params(Flux.params(NNR1s))
gs = gradient(ps) do
  loss(ss1,w,yy)
end

β0 = nothing
difNN = 1
for i in 1:100
    if i == 1
        data1 = (econsim0.sim[econsim0.sim[:,end].== 0,2:3], econsim0.sim[econsim0.sim[:,end].== 0,9])
        sst1,vrt1 = mydata(data1)
        traindatas = Flux.Data.DataLoader((sst1', vrt1'));
        vrhatNN, = NeuralEsti(NNR1s, traindatas, sst, vr)
        β0, = Flux.destructure(NNR1s)
    else
        ps = Flux.Params(Flux.params(NNR1s))
        gs = gradient(ps) do
          loss(sst1',vrt1')
        end
        Flux.update!(Descent(0.1),ps,gs)
        β1, = Flux.destructure(NNR1s)
        vrhatNN = ((1 / 2 * (NNR1s(sst')' .+ 1)) * (vrmax - vrmin) .+ vrmin)
        vrhatNN = reshape(vrhatNN, params.ne, params.nx)
        polfunS = update_solve(vrhatNN, polfun.vd, settings, params, uf)
        simaux = ModelSim(params, polfunS, settings, hf, nsim = Nsim)
        pdef = round(100 * sum(simaux.sim[:, 5]) / Nsim; digits = 2);
        data1 = (simaux.sim[simaux.sim[:,end].== 0,2:3], simaux.sim[simaux.sim[:,end].== 0,9])
        sst1,vrt1 = mydata(data1)
        difNN  = maximum(abs.(β0-β1))
        display("Simulation finished, with frequency of $pdef default events");
        display("Iteration $i, updating difference: $difNN");
    end
end
