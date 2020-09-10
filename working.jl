###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Distributions, Statistics, LinearAlgebra, Plots,
    StatsBase, Parameters, Flux;
include("supcodes.jl");
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
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt");
heatVR = heatmap(settings.y, settings.b, polfun.vr',
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt",
        c = :Accent_3);
#
############################################################
# Simulation
############################################################
econsim0 = ModelSim(params,polfun, settings,hf, nsim=100000);
pdef = round(100 * sum(econsim0.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

econsim1 = ModelSim(params,polfun, settings,hf, nsim=1000000);
pdef = round(100 * sum(econsim1.sim[:, 5])/ 1000000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

myunique(data) = begin
    dataTu = [Tuple(data[i,:])  for i in 1:size(data)[1]]
    dataTu = unique(dataTu)
    dataTu = [[dataTu[i]...]' for i in 1:length(dataTu)]
    data   = [vcat(dataTu...)][1]
    return data
end

data0 = myunique(econsim0.sim)
data1 = myunique(econsim1.sim)

preheatmap(data,params,settings)= begin
    sts = data[:,2:3]
    datav = data[:,5:6]
    DD  = -ones(params.ne,params.nx)
    VR  = fill(NaN,params.ne,params.nx)
    stuple = [(sts[i,1], sts[i,2])  for i in 1:size(sts)[1] ];
    for i in 1:params.ne
        bi = settings.b[i]
        for j in 1:params.nx
            yj = settings.y[j]
            pos = findfirst(x -> x==(bi,yj), stuple)
            if ~isnothing(pos)
                DD[i,j] = datav[pos, 1]
                VR[i,j] = datav[pos, 2]
            end
        end
    end
    return DD, VR
end

DD0, VR0 = preheatmap(data0,params,settings);
heat0 = heatmap(settings.y, settings.b, DD0', c = cgrad([:white, :black, :yellow]),
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
heat0VR = heatmap(settings.y, settings.b, VR0',aspect_ratio = 0.8, xlabel = "Output",
                ylabel = "Debt", c = :Accent_3 , grid=:false);

DD1, VR1 = preheatmap(data1,params,settings)
heat1 = heatmap(settings.y, settings.b, DD1', c = cgrad([:white, :black, :yellow]),
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
heat1VR = heatmap(settings.y, settings.b, VR1',aspect_ratio = 0.8, xlabel = "Output",
                ylabel = "Debt", c = :Accent_3 , grid=:false);

# Plotting HEATMAPS by sample size
plot(heat,heat0,heat1,layout = (3,1), size = (600,1200), title = ["(Actual)" "(100 m)" "(1 millon)"])
savefig("./Figures/heatmap_D.png");
plot(heatVR,heat0VR,heat1VR,layout = (3,1), size = (600,1200), title = ["(Actual)" "(100 m)" "(1 millon)"])
savefig("./Figures/heatmap_V.png");

############################################################
# Training
############################################################
mynorm(x) = begin
    ux = maximum(x, dims=1)
    lx = minimum(x, dims=1)
    nx = (x .- 0.5(ux+lx)) ./ (0.5*(ux-lx))
    return nx, ux, lx
end

ss0 = econsim0.sim[:,[2,8]]  # bₜ, yₜ
vr  = econsim0.sim[:,9]      # vᵣ
vr, uvr, lvr = mynorm(vr);
ss, uss, lss = mynorm(ss0);
data = (Array{Float32}(ss'), Array{Float32}(vr'));

mytrain(NN,data) = begin
    lossf(x,y) = Flux.mse(NN(x),y);
    traindata  = Flux.Data.DataLoader(data)
    pstrain = Flux.params(NN)
    Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, traindata, Descent())
end


# In-sample
    NNR1 = Chain(Dense(2, 16, softplus), Dense(16, 1));
    mytrain(NNR1,data);
    NNR2 = Chain(Dense(2, 16, tanh), Dense(16, 1));
    mytrain(NNR2,data);
    NNR3 = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
    mytrain(NNR3,data);
    NNR4 = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));
    mytrain(NNR4,data);
    uniqdata = unique([data[1]; data[2]],dims=2)
    fitt = [NNR1(uniqdata[1:2,:])' NNR2(uniqdata[1:2,:])' NNR3(uniqdata[1:2,:])' NNR4(uniqdata[1:2,:])']
    diff = abs.(uniqdata[3,:] .- fitt)
    scatter(uniqdata[1,:],uniqdata[2,:], [diff[:,1], diff[:,2], diff[:,3], diff[:,4]]);
# Out-sample
    bnorm = (settings.b .- 0.5(uss[1]+lss[1])) ./ (0.5*(uss[1]-lss[1]))
    ynorm = (settings.y .- 0.5(uss[2]+lss[2])) ./ (0.5*(uss[2]-lss[2]))
    states = [repeat(bnorm,params.nx)' ; repeat(ynorm,inner= (params.ne,1))']
    outpre = [NNR1(states)' NNR2(states)' NNR3(states)' NNR4(states)']



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

vr1 = (newsim[:,3] .- 0.5*(uvr + lvr)) ./ (0.5*(uvr - lvr));
ss1 = (newsim[:,1:2] .- 0.5*(uss + lss)) ./ (0.5*(uss - lss));
datar1 = Flux.Data.DataLoader((ss1',vr1'))
Flux.Optimise.train!(llvr, psr, datar1, Descent())
display(llvr(ss', vr'))
