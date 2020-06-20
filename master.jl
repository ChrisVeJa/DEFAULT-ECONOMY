using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");

EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
DefaultEconomy.graph_solve(EconSol);

EconSim = DefaultEconomy.ModelSimulate(EconSol);
DefaultEconomy.graph_simul(EconSim);

VFNeuF  = (vf= EconSim.Simulation[:,6], q = EconSim.Simulation[:,7],
			states= EconSim.Simulation[:,2:3]);
VFhat   = DefaultEconomy.neuralAprrox(VFNeuF[1],VFNeuF[3]);
DefaultEconomy.graph_neural(VFhat, "Value Function", ["VFneural.pdf" "VFNeuralSmpl.pdf"]);

ns         = 2;
Q          = 16;
ϕfun(x)    = log(1+exp(x));
mhat       = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
loss(x,y)  = Flux.mse(mhat(x),y);
opt        = RADAM();
NeuralChar = DefaultEconomy.NeuralSettings(mhat,loss,opt);
qhat       = DefaultEconomy.neuralAprrox(VFNeuF[2],VFNeuF[3],neuSettings=NeuralChar);
DefaultEconomy.graph_neural(qhat, "Bond price", ["BQneural.png" "BQNeuralSmpl.png"]);
