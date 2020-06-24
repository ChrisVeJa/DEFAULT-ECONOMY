###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
# The following code exemplifies the solution, simulation and approximation
# of a basic Arellano-type economy with default

# Written by: Christian Velasquez (velasqcb@bc.edu)
# Dont hesitate in send any comment
###############################################################################

# [0]
using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");

# [1]
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=100000,burn=0.05);
DefaultEconomy.graph_simul(EconSim);
VFNeuF  = (vf= EconSim.Simulation[:,6], q = EconSim.Simulation[:,7],states= EconSim.Simulation[:,2:3]);
ns      = 2; Q = 16;
ϕfun(x) = log(1+exp(x));
mhat_q  = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
loss(x,y)= Flux.mse(mhat_q(x),y);
opt     = RADAM();
NQChar  = DefaultEconomy.NeuralSettings(mhat_q,loss,opt);
VFhat   = DefaultEconomy.neuralAprrox(VFNeuF[1],VFNeuF[3]);
qhat    = DefaultEconomy.neuralAprrox(VFNeuF[2],VFNeuF[3],neuSettings=NQChar);

# [2]
normfun(x)           = (x .- 0.5*(maximum(x,dims=1)+minimum(x,dims=1))) ./ (0.5*(maximum(x,dims=1)-minimum(x,dims=1)));
norminv(x,xmax,xmin) = (x * 0.5*(xmax-xmin)) .+ 0.5*(xmax+xmin);
bgrid  = EconSol.Support.bgrid;
ygrid  = EconSol.Support.ygrid;
states = [repeat(bgrid,length(ygrid),1) repeat(ygrid,inner = (length(bgrid),1))];
sta_norm = normfun(states);
vfpre  = VFhat.mhat(sta_norm');
qpre   = qhat.mhat(sta_norm');
vfpre  = norminv(vfpre,maximum(VFNeuF.vf),minimum(VFNeuF.vf));
qpre   = norminv(qpre,maximum(VFNeuF.q),minimum(VFNeuF.q));
qpre   = max.(qpre,0);

vfΘ  = reshape(vfpre,length(bgrid),length(ygrid));
qΘ   = reshape(qpre,length(bgrid),length(ygrid));
