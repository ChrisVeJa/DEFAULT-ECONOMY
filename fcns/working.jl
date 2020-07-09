###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random, Distributions,Statistics, LinearAlgebra,
	Plots,StatsBase,Parameters, Flux;
include("DefEcon.jl");
include("convergence.jl");

############################################################
#[1] Setting >> Solving
#	As first step, I will set the model
#   in its defaults features
############################################################
Params, hdef, uf = DefEcon.ModelSettings();
EconDef = DefEcon.SolveR(Params, hdef, uf);
#DefEcon.graph_solve(EconDef);

############################################################
#	[Simulation]
############################################################
PF   = EconDef.PolFun;
Ext  = EconDef.Ext;
tsim = 100000;
tburn= 0.05;
EconSim = DefEcon.ModelSim(Params,PF,Ext,nsim = tsim, burn = tburn);
#DefEcon.graph_simul(EconSim, smpl=1:500);

############################################################
#	[Neural Network]
############################################################
vf= EconSim.Sim[:,6];
st= EconSim.Sim[:,2:3];
Q = 16;
n, ns= size(st);
ϕf(x)= log1p(exp(x));
mhat = Chain(Dense(ns,Q,ϕf), Dense(Q,1));
Loss(x,y) = Flux.mse(mhat(x),y);
opt  = Descent();
NseT = (mhat = mhat, loss = Loss,opt = opt);
norm = DefEcon.mynorm;
VFhat= DefEcon.NeuTra(vf,st,NseT,norm, Nepoch = 10);
#graph_neural(VFhat,"Value Function", ["VFneural.png" "VFSmpl.png"],smpl=1:500);

############################################################
# [2] Solving - simulating - training
############################################################
out1, out2= convergence(VFhat, NseT, Params, Ext, uf, tburn);
