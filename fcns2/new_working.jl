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
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 251);
uf(x, σrisk)= x.^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# Solving
############################################################
@time polfun, settings = Solver(params, hf, uf);
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

y  = econsim.sim[:,8];
ys, uys, lys = mynorm(y);
bs = econsim.sim[:,2];
ss = [-bs ys]
bb = -econsim.sim[:,4];
Q1    = 16
NNB   = Chain(Dense(2, Q1, softplus), Dense(Q1, 1, sigmoid))
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

newsim = Array{Float64,2}(undef,100*1000,4)
for i = 1:100
simaux = ModelSim(params,polfun, settings,hf, nsim=100000);
select = rand(1:100000,1000,1)
newsim[(i-1)*1000+1:i*1000,:] = simaux.sim[select,[8,2,4,5]]
end



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

updated!(v, vr,vd, D, uvr, udef, params, bp, P) = begin
    @unpack β, θ = params
    ev  = v * P'
    evd = vd * P'
    vr1 = uvr + β*ev[bp]
    evaux =  θ * ev[end, :]' .+  (1 - θ) * evd
    vd1 = udef' .+ β*evaux
    v1  = max.(vr1,vd1)
    dif = max(maximum(abs.(v1-v)),maximum(abs.(vr1-vr)), maximum(abs.(vd1-vd)))
    return v1, vr1, vd1, dif
end

polfun.
vg,vrg,vdg, iteration = fixedp(bb,D,bp, settings, params, uf)
display(maximum(abs.(vdg-polfun.vd)))
