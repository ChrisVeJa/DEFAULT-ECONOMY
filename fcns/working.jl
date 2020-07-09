###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,
    Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux;
include("DefEcon.jl");
include("convergence.jl");

############################################################
#[1] Setting >> Solving
#	As first step, I will set the model
#   in its defaults features
############################################################
Params, hdef, uf = DefEcon.ModelSettings();
Params = (
    r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);
EconDef = DefEcon.SolveR(Params, hdef, uf);
PF   = EconDef.PolFun;
Ext  = EconDef.Ext;
DefEcon.graph_solve(Params,PF,Ext);
############################################################
#	[Simulation]
############################################################
tsim = 100000;
tburn = 0.05;
EconSim = DefEcon.ModelSim(Params, PF, Ext, nsim = tsim, burn = tburn);
DefEcon.graph_simul(EconSim, smpl=1:500);

############################################################
#	[Neural Network]
############################################################
# I will estimated  two neural networks:
#        i) for no default
#       ii) for default
Q  = 16; ns = 2; ϕf(x) = log1p(exp(x)); opt = Descent();
norm = DefEcon.mynorm;
vf = EconSim.Sim[:, 6];
st = EconSim.Sim[:, 2:3];
defstatus =  EconSim.Sim[:, 5];

vnd = vf[defstatus .== 0]
snd = st[defstatus .== 0, :]
mnDef = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Lnd(x, y) = Flux.mse(mnDef(x), y);
NseTnD = (mhat = mnDef, loss = Lnd, opt = opt);
VNDhat = DefEcon.NeuTra(vnd, snd, NseTnD, norm, Nepoch = 10);
DefEcon.graph_neural(VNDhat,"Value Function No Default", ["VNDneural.png" "VNDSmpl.png"],smpl=1:500);

vd = vf[defstatus .== 1]
sd = st[defstatus .== 1, :]
mDef = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Ld(x, y) = Flux.mse(mDef(x), y);
NseTD = (mhat = mDef, loss = Ld, opt = opt);
VDhat = DefEcon.NeuTra(vd, sd, NseTD, norm, Nepoch = 10);
DefEcon.graph_neural(VDhat,"Value Function Default", ["VDneural.png" "VDSmpl.png"],smpl=1:200);

############################################################
# [2] Solving - simulating - training
############################################################
PolFun1, EconSim1= convergence(VNDhat,VDhat, NseT, Params, Ext, uf, tburn);
VNDhat.
