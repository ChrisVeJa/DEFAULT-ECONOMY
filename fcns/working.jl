###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
# The following code exemplifies the solution, simulation and approximation
# of a basic Arellano-type economy with default

# Written by: Christian Velasquez (velasqcb@bc.edu)
# Dont hesitate in send any comment
###############################################################################

# [0] Including our module
using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");

# Setting >> Solving >> Simulating >> Neural Network
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=10000,burn=0.05);
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);


# [4] Solving - Simulating - Training
q = EconSim.Sim[:,6];
VFNeuFAux, VFhatAux, qhatAux, d1 = DefaultEconomy.ConvergeNN(EconSol,
									VFNeuF,VFhat,q, flagq=false, nrep=10000, maxite=2000)



#d1, seed fixed, descent, 10000 rep, 0.9
plot(d1, title= "10milFiXedDescent90");
savefig("graph1.png");
