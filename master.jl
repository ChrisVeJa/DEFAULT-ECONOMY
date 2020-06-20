###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
# The following code exemplifies the solution, simulation and approximation
# of a basic Arellano-type economy with default

# Written by: Christian Velasquez (velasqcb@bc.edu)
# Dont hesitate in send any comment
###############################################################################

# Including out module
using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");

# Settings and Solving the basic model
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
DefaultEconomy.graph_solve(EconSol);

# Simulating the model
EconSim = DefaultEconomy.ModelSimulate(EconSol);
DefaultEconomy.graph_simul(EconSim);

# Approximation with Neural Networks
  # Value Function
  	VFNeuF  = (vf= EconSim.Simulation[:,6], q = EconSim.Simulation[:,7],
			states= EconSim.Simulation[:,2:3]);
	VFhat   = DefaultEconomy.neuralAprrox(VFNeuF[1],VFNeuF[3]);
	DefaultEconomy.graph_neural(VFhat, "Value Function", ["VFneural.png" "VFNeuralSmpl.png"]);

  # Bond Prfice
	ns         = 2;
	Q          = 16;
	ϕfun(x)    = log(1+exp(x));
	mhat       = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
	loss(x,y)  = Flux.mse(mhat(x),y);
	opt        = Descent();
	NeuralChar = DefaultEconomy.NeuralSettings(mhat,loss,opt);
	qhat       = DefaultEconomy.neuralAprrox(VFNeuF[2],VFNeuF[3],neuSettings=NeuralChar);
	DefaultEconomy.graph_neural(qhat, "Bond price", ["BQneural.png" "BQNeuralSmpl.png"]);
