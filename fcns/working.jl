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
include("convergenceNN.jl");

# Setting >> Solving >> Simulating >> Neural Network
totalsim = 10000;
totalburn= 0.05;
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=totalsim,burn=totalburn);
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);

# [4] Solving - Simulating - Training
EconSimF = convergenceNN(EconSol,VFhat, totalburn);



DefaultEconomy.graph_solve(EconSol);
DefaultEconomy.graph_solve(EconSol1);























#=VFNeuFAux, VFhatAux, qhatAux, d1 = DefaultEconomy.ConvergeNN(EconSol,
#									VFNeuF,VFhat,q, qtype="NoUpdate", nrep=10000, maxite=4000)
#d1, seed fixed, descent, 10000 rep, 0.9

VFNeuFAux, VFhatAux, qhatAux, d2 = DefaultEconomy.ConvergeNN(EconSol,
									VFNeuF,VFhat,q, qtype="UpdateActual", nrep=totalsim, maxite=4)


plot([d1 d2], title= "10milFiXedDescent90")
savefig("graph3.png");
=#
