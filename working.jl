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

# [1] Solving the model
EconDef = DefaultEconomy.ModelSettings();
EconSol = DefaultEconomy.SolveDefEcon(EconDef);

# [2] Simulate a sample with 1e5 observations
EconSim = DefaultEconomy.ModelSimulate(EconSol,nsim=100000,burn=0.05);

# [3] Estimating a Neural network
    # 3.1 Data for training
    VFNeuF  = (vf= EconSim.Sim[:,6], q = EconSim.Sim[:,7],states= EconSim.Sim[:,2:3]);

    # 3.2 Neural network for Value Funtion 16 neurons, 1 hidden layer, with softplus
    VFhat   = DefaultEconomy.NeuralTraining(VFNeuF[1],VFNeuF[3], Nepoch = 10);
    # 3.3 Neural network for Bond Price  16 neurons, 2 hidden layer, with softplus
    ns      = 2; Q = 16;
    ϕfun(x) = log1p(exp(x));
    #mhat_q  = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
    mhat_q  = Chain(Dense(ns,Q,ϕfun),Dense(Q,Q,ϕfun), Dense(Q,1));
    loss(x,y)= Flux.mse(mhat_q(x),y);
    opt     = Descent();
    NQChar  = DefaultEconomy.NeuralSettings(mhat_q,loss,opt);
    qhat    = DefaultEconomy.NeuralTraining(VFNeuF[2],VFNeuF[3],neuSettings=NQChar,Nepoch = 10);


# [4] Solving - Simulating - Training
    VFNeuFAux, VFhatAux, qhatAux, DIF= DefaultEconomy.ConvergeNN(EconSol,VFNeuF,VFhat,qhat);
    Γ1old, re11   = Flux.destructure(VFhat.Mhat);
	Γ2old, re21   = Flux.destructure(qhat.Mhat);
	Γ1oldA, re11A = Flux.destructure(VFhatAux.Mhat);
	Γ2oldA, re21A = Flux.destructure(qhatAux.Mhat);
