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
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=100000,burn=0.05);
VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);
VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);


# [4] Solving - Simulating - Training
q = EconSim.Sim[:,6];
VFNeuFAux, VFhatAux, qhatAux, DIF = DefaultEconomy.ConvergeNN(EconSol,VFNeuF,VFhat,q, flagq=false, nrep=100000)
p3 = plot(DIF)









Γ1old, re11   = Flux.destructure(VFhat.Mhat);
Γ2old, re21   = Flux.destructure(qhat.Mhat);
Γ1oldA, re11A = Flux.destructure(VFhatAux.Mhat);
Γ2oldA, re21A = Flux.destructure(qhatAux.Mhat);
plot(DIF, legend= false)
lens!([200, 400], [0.1, 0.3], inset = (1, bbox(0.5, 0.0, 0.4, 0.4)))
