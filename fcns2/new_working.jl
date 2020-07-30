###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Optim, Distributions, Statistics, LinearAlgebra, Plots,
    StatsBase, Parameters, Flux
include("supcodes.jl")
############################################################
# []  Functions
############################################################

############################################################
# SETTING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 501);
uf(x, σrisk)= x.^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# Solving
############################################################
polfun, settings = Solver(params, hf, uf);
heat = heatmap(settings.y, settings.b, polfun.D',
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
#savefig("./Figures/heatmap_base.svg")
############################################################
# Simulation
############################################################
econsim = ModelSim(params,polfun, settings,hf, nsim=100000);
pdef = round(100 * sum(econsim.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

DD  = -ones(params.ne,params.nx)
sts =  econsim.sim[:,2:3]
stuple = [(sts[i,1], sts[i,2])  for i in 1:size(sts)[1] ];
for i in 1:params.ne
    bi = settings.b[i]
    for j in 1:params.nx
        yj = settings.y[j]
        pos = findfirst(x -> x==(bi,yj), stuple)
        if ~isnothing(pos)
            DD[i,j] = econsim.sim[pos, 5]
        end
    end
end
heat1 = heatmap(settings.y, settings.b, DD', c = cgrad([:white, :black, :yellow]),aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );

#savefig("./Figures/compheat_sim.svg")

############################################################
# Training
############################################################
mynorm(x) = begin
    ux = maximum(x, dims=1)
    lx = minimum(x, dims=1)
    nx = (x .- 0.5(ux+lx)) ./ (0.5*(ux-lx))
    return nx, ux, lx
end

ss0 = econsim.sim[:,[2,8]] # bₜ, yₜ
vr = econsim.sim[:,9] # bₜ, yₜ
vd = econsim.sim[:,10];
vr, uvr, lvr = mynorm(vr);
vd, uvd, lvd = mynorm(vd);
ss, uss, lss = mynorm(ss0);
Q1 = 16
NNR  = Chain(Dense(2, Q1, softplus), Dense(Q1, 1));
llvr(x,y)  = Flux.mse(NNR(x),y);
datar = Flux.Data.DataLoader((ss',vr'))
psr = Flux.params(NNR)
Flux.@epochs 10 begin
   Flux.Optimise.train!(llvr, psr, datar, Descent())
   display(llvr(ss', vr'))
end

# Fit
vrfit = NNR(ss')'
datafit = [(ss0[i,1], ss0[i,2], vrfit[i] ,vr[i])  for i in eachindex(vr)]
datafit = unique(datafit)
datafit = [[datafit[i][1] datafit[i][2] datafit[i][3] datafit[i][4]] for i in 1:length(datafit)]
datafit = [vcat(datafit...)][1]
datafit = sort(datafit, dims=1)
diff = abs.(datafit[:,4] - datafit[:,3])
sc1 = scatter(datafit[:,2],datafit[:,1],diff, marker_z = (+),markersize = 5,
        color = :bluesreds, label ="", markerstrokewidth = 0.1)




NNR1  = Chain(Dense(2, Q1, tanh), Dense(Q1, 1));
llvr1(x,y)  = Flux.mse(NNR1(x),y);
datar1 = Flux.Data.DataLoader((ss',vr'))
psr1 = Flux.params(NNR1)
Flux.@epochs 10 begin
   Flux.Optimise.train!(llvr1, psr1, datar1, Descent())
   display(llvr1(ss', vr'))
end

# Fit
vrfit1 = NNR1(ss')'
datafit1 = [(ss0[i,1], ss0[i,2], vrfit1[i] ,vr[i])  for i in eachindex(vr)]
datafit1 = unique(datafit1)
datafit1 = [[datafit1[i][1] datafit1[i][2] datafit1[i][3] datafit1[i][4]] for i in 1:length(datafit1)]
datafit1 = [vcat(datafit1...)][1]
datafit1 = sort(datafit1, dims=1)
diff1 = abs.(datafit1[:,4] - datafit1[:,3])
sc2 = scatter(datafit1[:,2],datafit1[:,1],diff1, marker_z = (+),markersize = 5,
        color = :bluesreds, label ="", markerstrokewidth = 0.1)


NNR1  = Chain(Dense(2, Q1, relu), Dense(Q1, Q1,tanh), Dense(Q1,1));
llvr1(x,y)  = Flux.mse(NNR1(x),y);
datar1 = Flux.Data.DataLoader((ss',vr'))
psr1 = Flux.params(NNR1)
Flux.@epochs 10 begin
   Flux.Optimise.train!(llvr1, psr1, datar1, Descent())
   display(llvr1(ss', vr'))
end

# Fit
vrfit1 = NNR1(ss')'
datafit1 = [(ss0[i,1], ss0[i,2], vrfit1[i] ,vr[i])  for i in eachindex(vr)]
datafit1 = unique(datafit1)
datafit1 = [[datafit1[i][1] datafit1[i][2] datafit1[i][3] datafit1[i][4]] for i in 1:length(datafit1)]
datafit1 = [vcat(datafit1...)][1]
datafit1 = sort(datafit1, dims=1)
diff1 = abs.(datafit1[:,4] - datafit1[:,3])
sc3 = scatter(datafit1[:,2],datafit1[:,1],diff1, marker_z = (+),markersize = 5,
        color = :bluesreds, label ="", markerstrokewidth = 0.1)



NNR1  = Chain(Dense(2, Q1, relu), Dense(Q1, Q1,softplus), Dense(Q1,1));
llvr1(x,y)  = Flux.mse(NNR1(x),y);
datar1 = Flux.Data.DataLoader((ss',vr'))
psr1 = Flux.params(NNR1)
Flux.@epochs 10 begin
  Flux.Optimise.train!(llvr1, psr1, datar1, Descent())
  display(llvr1(ss', vr'))
end

# Fit
vrfit1 = NNR1(ss')'
datafit1 = [(ss0[i,1], ss0[i,2], vrfit1[i] ,vr[i])  for i in eachindex(vr)]
datafit1 = unique(datafit1)
datafit1 = [[datafit1[i][1] datafit1[i][2] datafit1[i][3] datafit1[i][4]] for i in 1:length(datafit1)]
datafit1 = [vcat(datafit1...)][1]
datafit1 = sort(datafit1, dims=1)
diff1 = abs.(datafit1[:,4] - datafit1[:,3])
sc4 = scatter(datafit1[:,2],datafit1[:,1],diff1, marker_z = (+),markersize = 5,
       color = :bluesreds, label ="", markerstrokewidth = 0.1)











newsim = Array{Float64,2}(undef,100*1000,4)
for i = 1:10
    simaux = ModelSim(params,polfun, settings,hf, nsim=100000);
    newsim[(i-1)*10000+1:i*10000,:] = simaux.sim[rand(1:100000,10000,1),[2,8,9,10]]
end

vr1 = (newsim[:,3] .- 0.5*(uvr + lvr)) ./ (0.5*(uvr - lvr));
ss1 = (newsim[:,1:2] .- 0.5*(uss + lss)) ./ (0.5*(uss - lss));
datar1 = Flux.Data.DataLoader((ss1',vr1'))
Flux.Optimise.train!(llvr, psr, datar1, Descent())
display(llvr(ss', vr'))



NND  = Chain(Dense(2, 4, softplus), Dense(4, 1))
lossb(x, y) = Flux.mse(NNB(x), y)
datab = Flux.Data.DataLoader(ss', bb')
psb   = Flux.params(NNB)
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossb, psb, datab, Descent())
   display(lossb(ss', bb'))
end
dplot = [bb NNB(ss')'];
p1 = plot(dplot[1:500,:], label = ["bond" "NN"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red], alpha = 0.7, w = [1.15 0.75], legend=:topright, grid=:false);


dd = econsim.sim[:,5];
Q1 = 32
NND = Chain(Dense(2,Q1,softplus), Dense(Q1,1,sigmoid));
lossd(x::Array,y::Array) = begin
     x1 = NND(x)
     logll = log.(x1 .+ 0.0000001).*y + log.(1.0000001 .- x1) .* (1 .-y)
     logll = mean(logll)
     return -logll
end
ss1   = Array{Float32}(ss')
dd1   = Array{Float32}(dd')
datad = Flux.Data.DataLoader(ss1,dd1)
psd   = Flux.params(NND);
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossd, psd, datad, ADAM())
   display(lossd(ss1, dd1))
end
dhat1 = NND(ss1)'
dplot2 = [dd dhat1];
p2 = plot(dplot2[1:5000,:],label = ["default" "NN"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red], alpha = 0.7, w = [1.15 0.75], legend=:topright, grid=:false);





y1sim  = newsim[:,1];  ys1sim, uys1sim, lys1sim = mynorm(y1sim);
bs1sim = newsim[:,2];
sssim  = [-bs1sim ys1sim]
ss1sim = Array{Float32}(sssim')
dd1sim = Array{Float32}(newsim[:,4]')
datadsim = Flux.Data.DataLoader(ss1sim,dd1sim)
Flux.Optimise.train!(lossd, psd, datadsim, ADAM())
dhat2 = NND(ss1sim)'
dplot3 = [dd dhat1 dhat2];
p3 = plot(dplot3[1:5000,[1,3]],label = ["default" "NN" "NN1"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red :purple], alpha = 0.7, w = [1.15 0.75 0.75], legend=:topright, grid=:false);







function train!(loss, ps, data, opt; cb = () -> ())
  ps = Params(ps)
  @progress for d in data
      gs = gradient(ps) do
        loss(batchmemaybe(d)...)
      update!(opt, ps, gs)
      cb()
end





D = polfun.D
bb = polfun.bb
bp = polfun.bp

fixedp(bb,D,bp, settings, params, uf) = begin
    @unpack P, y, b,udef   = settings
    @unpack σrisk, β, ne,r, θ = params
    vd = 1/(1-β) * udef;
    vd = repeat(vd',ne, 1)
    vr = 1/(1 - β) * uf.((r / (1 + r)) * b .+ y', σrisk)
    q  = (1 / (1 + r)) * (1 .- (D*P'))
    v   = max.(vr,vd)
    cc  = (b .+ y') - (q[bp].*bb);
    cc[cc.<0] .= 0;
    uvr = uf.(cc,σrisk)
    dif = 1.0
    iteration = 1
    while dif > 1e-8 && iteration<10000
        v1, vr1, vd1 , dif = updated!(v,vr, vd,D, uvr, udef,params,bp, P)
        vr = vr1; vd=  vd1 ; v = v1;
        iteration+=1
    end
    return v,vr,vd, iteration
end
