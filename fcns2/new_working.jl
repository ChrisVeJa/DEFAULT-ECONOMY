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

mytrain(NN,data,ss,vr; col = :blue) = begin
    lossf(x,y) = Flux.mse(NN(x),y);
    traindata  = Flux.Data.DataLoader(data)
    pstrain = Flux.params(NN)
    Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, traindata, Descent())
    vrfit = NN(data[1])'
    fit = [(ss[i,1], ss[i,2], vrfit[i] ,vr[i])  for i in eachindex(vr)]
    fit = unique(fit)
    fit = [[fit[i][1] fit[i][2] fit[i][3] fit[i][4]] for i in 1:length(fit)]
    fit = [vcat(fit...)][1]
    fit = sort(fit, dims=1)
    diff = abs.(fit[:,4] - fit[:,3])
    sc = scatter(fit[:,2], fit[:,1],diff,markersize = 5, legend = :false,
            color = col, alpha =0.2, label ="", markerstrokewidth = 0.1);
    return sc, fit[:,2],fit[:,1],diff
end

ss0 = econsim.sim[:,[2,8]] # bₜ, yₜ
vr = econsim.sim[:,9]      # bₜ, yₜ
vr, uvr, lvr = mynorm(vr);
ss, uss, lss = mynorm(ss0);
data = (Array{Float32}(ss'), Array{Float32}(vr'));
NNR1 = Chain(Dense(2, 16, softplus), Dense(16, 1));
sc1  = mytrain(NNR1,data,ss0,vr,col = :orange);
NNR2 = Chain(Dense(2, 16, tanh), Dense(16, 1));
sc2  = mytrain(NNR2,data,ss0,vr,col = :sienna4);
NNR3 = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
sc3  = mytrain(NNR3,data,ss0,vr,col = :purple);
NNR4 = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));
sc4  = mytrain(NNR4,data,ss0,vr,col = :teal);
tit = ["Softplus" "Tanh" "Relu + softplus" "Relu + tanh"]
plot(sc1[1],sc2[1],sc3[1],sc4[1],layout = (2,2),size=(1000,800),
    title = tit)

scatter(sc1[2],sc1[3],[sc1[4], sc2[4],sc3[4],sc4[4]], alpha =0.2,
    label = tit, legendfontsize = 8, fg_legend = :transparent,
    bg_legend = :transparent, legend = :topleft,size=(800,600))

difmat = [sc1[3] sc1[2] sc1[4] sc2[4] sc3[4] sc4[4]];
difmat = sort(difmat,dims=1)
difmat = difmat[:,3:end]
minvalmat = [findmin(difmat[i,:])[1] for i in 1:size(difmat)[1]]
minmat = [findmin(difmat[i,:])[2] for i in 1:size(difmat)[1]]
minarc = tit[minmat];
bestNN = [sort([sc1[3] sc1[2]],dims=1) minvalmat difmat minarc]

io = open("bestNN.txt", "w");
for i in 1:size(bestNN)[1]
    for j in 1:7
        print(io, round(bestNN[i,j], digits=4),"\t")
    end
    println(io,bestNN[i,8])
end
close(io);

polsta = [ repeat(settings.b, params.nx) repeat(settings.y, inner = (params.ne,1))];
pred1 = NNR1(polsta')';
pred1 = ((pred1) .* (0.5*(uvr-lvr))) .+ (0.5*(uvr+lvr))
pred1 = reshape(pred1, params.ne,params.nx);
difpre1 = pred1 - polfun.vr;
pred2 = NNR2(polsta')';
pred2 = ((pred2) .* (0.5*(uvr-lvr))) .+ (0.5*(uvr+lvr))
pred2 = reshape(pred2, params.ne,params.nx);
difpre2 = pred2 - polfun.vr;
pred3 = NNR3(polsta')';
pred3 = ((pred3) .* (0.5*(uvr-lvr))) .+ (0.5*(uvr+lvr))
pred3 = reshape(pred3, params.ne,params.nx);
difpre3 = pred3 - polfun.vr;
pred4 = NNR4(polsta')';
pred4 = ((pred4) .* (0.5*(uvr-lvr))) .+ (0.5*(uvr+lvr))
pred4 = reshape(pred4, params.ne,params.nx);
difpre4 = pred4 - polfun.vr;

anim = @animate for i in 1:21
plot(settings.b,[difpre1[:,i] difpre2[:,i] difpre3[:,i] difpre4[:,i]],
    c= [:black :green :blue :red], grid= :false, xlabel = "debt",
    label= ["softplus" "tanh" "relu+softplus" "relu+tanh"],
    fg_legend = :transparent, bg_legend = :transparent,
    legend_title = "state for y = $i", legendtitlefontsize = 8,
    style = [:solid :solid :dash :dash],
    title = "VR predicted - VR actual")
end
gif(anim,"myanim.gif",fps = 1)

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
