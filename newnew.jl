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
par = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
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
@time polfun, set = Solver(par, hf, uf);
MoDel = hcat([vec(i) for i in polfun]...);
MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) MoDel];
heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q, :bb];
ModelData = DataFrame(Tables.table(MoDel, header = heads));
yticks = round.(set.y, digits = 2);
yticks = [yticks[1], yticks[6], yticks[11], yticks[16], yticks[end]];
heat0 = Gadfly.plot(ModelData, x = "debt", y = "output", color = "D", Geom.rectbin,
    Scale.color_discrete_manual("red","green"), Theme(background_color = "white"),
    Theme(background_color = "white",
            key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"), Guide.yticks(ticks = yticks),
    Guide.colorkey(title = "Default choice", labels = ["No Default", "Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]))
# ---------------------------------------------------
# [3.b] Simulating data from  the model
# ----------------------------------------------------------
Nsim = 100_000
sim0 = ModelSim(par, polfun, set, hf, nsim = Nsim);
pdef = round(100 * sum(sim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

mataux = dropdims(sim0.sim[:,[2 8 5 9]], dims=2); # we pick only y_j
mataux = unique(mataux,dims=1);
heads = [:debt, :output, :D, :vr];
MatAux= [MoDel[:,1:2] fill(NaN,size(MoDel,1),2)]

for i in 1:size(mataux,1)
    l1 = findfirst(x -> x == mataux[i, 1], set.b)
    l2 = findfirst(x -> x == mataux[i, 2], set.y)
    MatAux[(l2-1)*par.ne+l1,3:4] = mataux[i,3:4]
end

MatAux = DataFrame(Tables.table(MatAux, header = heads));
sort!(MatAux, :D)
heat1 = Gadfly.plot(MatAux, x = "debt", y = "output", color = "D", Geom.rectbin,
    Scale.color_discrete_manual("green", "red","white"), Theme(background_color = "white"),
    Theme(background_color = "white",
            key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"), Guide.yticks(ticks = yticks),
    Guide.colorkey(title = "Default choice", labels = ["No Default", "Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]))
############################################################
# [4] NEURAL NETWORK
############################################################

# ---------------------------------------------------
# [4.a] The model
# ---------------------------------------------------
dNN=16
NN = Chain(Dense(2, dNN, softplus), Dense(dNN, 1));


# ---------------------------------------------------
# [4.b] The data
# ---------------------------------------------------
x = [sim0.sim[sim0.sim[:,end].== 0,2:3] sim0.sim[sim0.sim[:,end].== 0,9]]
lx = minimum(x,dims=1)
ux = maximum(x,dims=1)
x0 = 2 * (x .- lx) ./ (ux - lx) .- 1
x0 = [x0[:,1:2] x[:,3]]
x0Tu = [Tuple(x0[i, :]) for i = 1:size(x0, 1)]
x1Tu = unique(x0Tu)
xx = countmap(x0Tu)
x1 = Array{Float32,2}(undef,size(x1Tu)[1],3)
for i in 1:size(x1Tu)[1]
    x1[i,:] = [x1Tu[i]...]'
end
ww  = [get(xx, i, 0) for i in x1Tu]
ss  = x1[:,1:2]
yy  = x1[:,3];

# ---------------------------------------------------
# [4.c] Estimation
# ---------------------------------------------------
# function for line search
loss(x,w,y,NN) = sum(w .* ((NN(x) - y).^2))/(sum(w))
lossbls(x,w,y,NN, θ) = begin
    be, _N0 = Flux.destructure(NN)
    NN1 = _N0(θ)
    loss = sum(w .* ((NN1(x) - y).^2))/(sum(w))
    return loss
end

_updateNN(NN,loss,lossbls,ss,ww,yy)= begin
    ps = Flux.Params(Flux.params(NN));
    gs = gradient(ps) do
      loss(ss',ww',yy',NN)
    end # gradient, it needs to be recalculated in each iteration
    #Backtracking line search
    θ, re =  Flux.destructure(NN)
    loss0 = loss(ss',ww',yy',NN)
    ∇θ = vcat(vec.([gs[i] for i in ps])...)
    ∇f0 = sum(∇θ.^2)
    α=1
    decay = 0.3
    while lossbls(ss',ww',yy',NN, θ-α.*∇θ) > loss0 - 0.5*α*∇f0
        α *= decay
    end
    θ1 = θ-α.*∇θ
    θ1 = 0.9*θ1 + 0.1*θ
    dif = maximum(abs.(θ1-θ))
    NN1 = re(θ1)
    return NN1, dif
end

_estNN(NN,loss,ss,ww,yy, lossbls) = begin
    dif = 1
    rep = 0
    NN1 = nothing
    while dif>1e-5 && rep<5000
        NN1, dif = _updateNN(NN,loss,lossbls,ss,ww,yy);
        NN = NN1;
        rep +=1;
    end
    display("Iteration $rep: maximum dif $dif")
    return NN1
end

θs , more = Flux.destructure(NN)
NN1 = nothing
NNopt = nothing
lossvalue = 1e10
for j in 1:10
    θaux = -1 .+ 2*rand(length(θs))
    NNaux = more(θaux)
    NN1 = _estNN(NNaux,loss,ss,ww,yy, lossbls);
    lossaux = loss(ss',ww',yy',NN1)
    display("Loss value of model $j is $lossaux")
    if lossaux < lossvalue
        NNopt = NN1
        lossvalue = lossaux
    end
end

# Projection over the whole grid
ssgrid = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))]
ssgrid = 2 * (ssgrid .- lx[1:2]') ./ (ux[1:2]' - lx[1:2]') .- 1
vrhat  =  reshape(NNopt(ssgrid'),par.ne,par.nx)
hat_vd = polfun.vd
polfunSfit = update_solve(vrhat, hat_vd, set, par, uf)


MoDel = hcat([vec(i) for i in polfunSfit]...);
MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) MoDel];
heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q, :bb];
ModelData = DataFrame(Tables.table(MoDel, header = heads));
yticks = round.(set.y, digits = 2);
yticks = [yticks[1], yticks[6], yticks[11], yticks[16], yticks[end]];
heat0s = Gadfly.plot(ModelData, x = "debt", y = "output", color = "D", Geom.rectbin,
    Scale.color_discrete_manual("red","green"), Theme(background_color = "white"),
    Theme(background_color = "white",
            key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"), Guide.yticks(ticks = yticks),
    Guide.colorkey(title = "Default choice", labels = ["No Default", "Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]))








simfit = ModelSim(par, polfunSfit, set, hf, nsim = Nsim)
pdef1 = round(100 * sum(simfit.sim[:, 5]) / Nsim; digits = 2)
display("Simulation finished $pdef1 percent")

mataux = dropdims(simfit.sim[:,[2 8 5 9]], dims=2); # we pick only y_j
mataux = unique(mataux,dims=1);
heads = [:debt, :output, :D, :vr];
MatAux= [MoDel[:,1:2] fill(NaN,size(MoDel,1),2)]

for i in 1:size(mataux,1)
    l1 = findfirst(x -> x == mataux[i, 1], set.b)
    l2 = findfirst(x -> x == mataux[i, 2], set.y)
    MatAux[(l2-1)*par.ne+l1,3:4] = mataux[i,3:4]
end

MatAux = DataFrame(Tables.table(MatAux, header = heads));
sort!(MatAux, :D)
heat2 = Gadfly.plot(MatAux, x = "debt", y = "output", color = "D", Geom.rectbin,
    Scale.color_discrete_manual("green", "red","white"), Theme(background_color = "white"),
    Theme(background_color = "white",
            key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"), Guide.yticks(ticks = yticks),
    Guide.colorkey(title = "Default choice", labels = ["No Default", "Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]))













vr     = vec(polfun.vr)
ssmin = minimum(ss, dims = 1);   ssmax = maximum(ss, dims = 1)
vrmin = minimum(vr) ;vrmax = maximum(vr)
sst = 2 * (ss .- ssmin) ./ (ssmax - ssmin) .- 1
vrt = 2 * (vr .- vrmin) ./ (vrmax - vrmin) .- 1

f1(x) = sqrt(mean(x .^ 2))                     # Square root of Mean Square Error
f2(x) = maximum(abs.(x))                       # Maximum Absolute Deviation
f3(x, y) = sqrt(mean((x ./ y) .^ 2)) * 100     # Square root of Mean Relative Square Error
f4(x, y) = maximum(abs.(x ./ y)) * 100         # Maximum Relative deviation
