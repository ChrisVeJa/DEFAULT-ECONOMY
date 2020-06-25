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
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
ns      = 2; Q = 16;
ϕfun(x) = log(1+exp(x));
mhat_q  = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
loss(x,y)= Flux.mse(mhat_q(x),y);
opt     = RADAM();
NQChar  = DefaultEconomy.NeuralSettings(mhat_q,loss,opt);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 1);
qhat    = DefaultEconomy.NeuralTraining(VFNeuF[2],VFNeuF[3],neuSettings=NQChar,Nepoch = 10);


# [2]
DefaultEconomy.lalaland(EconSol,VFNeuF,VFhat,qhat)
