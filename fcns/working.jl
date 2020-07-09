###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random, Distributions,Statistics, LinearAlgebra,
	Plots,StatsBase,Parameters, Flux;
include("DefEcon.jl");
#include("convergence.jl");
include("graphs.jl");

############################################################
#[1] Setting >> Solving
#	As first step, I will set the model
#   in its defaults features
############################################################
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
#graph_solve(EconSol);

############################################################
#	[Simulation]
############################################################
tsim    = 10000; tburn   = 0.05;
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=tsim,burn=tburn);
#graph_simul(EconSim, smpl=1:500);

############################################################
#	[Neural Network]
############################################################
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);
#graph_neural(VFhat,"Value Function", ["VFneural.png" "VFSmpl.png"],smpl=1:500);

############################################################
# [2] Solving - simulating - training
############################################################
convergenceNN(EconSol,VFhat, tburn);
