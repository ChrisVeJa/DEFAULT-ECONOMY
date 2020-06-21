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
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=100000,burn=0.05);
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
	opt        = RADAM();
	NeuralChar = DefaultEconomy.NeuralSettings(mhat,loss,opt);
	qhat       = DefaultEconomy.neuralAprrox(VFNeuF[2],VFNeuF[3],neuSettings=NeuralChar);
	DefaultEconomy.graph_neural(qhat, "Bond price", ["BQneural.png" "BQNeuralSmpl.png"]);

###################################
# ADDITIONAL FIGURES
##################################
q1    = qhat.Data[1];
s1    = qhat.Data[2];
q1hat = qhat.yhat;

# SCATTERS
theme(:ggplot2)
scatter(s1[:,1],[q1 q1hat], lab=["actual" "hat"],
		marker = [:circ :+], markersize = [3.25 3], c= [RGB(0.39,0.58,0.93) :red],
		title="Bond price vs Debt", legend = :bottomleft,
		xlab ="Debt Bₜ", ylab="Debt Price qₜ₊₁");
savefig(".\\FiguresAdi\\QB.png");
scatter(s1[:,2],[q1 q1hat], lab=["actual" "hat"],
		marker = [:circ :+], markersize = [3.25 3], c= [RGB(0.39,0.58,0.93) :red],
		title="Bond price vs Output",legend = :bottomleft,
		xlab ="Output Yₜ", ylab="Debt Price qₜ₊₁");
savefig(".\\FiguresAdi\\QY.png");

# 3D
theme(:default)
surf1 = surface(s1[:,2],s1[:,1],q1,c=:OrRd_9,
				xlab = "Bₜ", ylab ="yₜ", zlab="Actual Price")
surf2 = surface(s1[:,2],s1[:,1],q1hat,c=:OrRd_9,
				xlab = "Bₜ", ylab ="yₜ", zlab="Predicted Price")

plot(surf1,surf2, layout=(1,2),size = (900,400))
savefig(".\\FiguresAdi\\surface.png");
