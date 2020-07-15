###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,
    Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux;
include("DefEcon.jl");
include("graphs.jl");
############################################################
#[1] Setting >> Solving
############################################################
Params, hdef, uf = DefEcon.ModelSettings();
Params = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);

EconDef = DefEcon.SolveR(Params, hdef, uf);
tsim    = 100000;
tburn   = 0.05;
EconSim = DefEcon.ModelSim(Params, EconDef.PolFun, EconDef.Ext, nsim = tsim, burn = tburn);
NDef    = sum(EconSim.Sim[:,5]);
PDef    = round(100*NDef/ tsim; digits = 2);
display("Simulation finished, with $NDef defaults event and a frequency of $PDef")


ϕf(x)= log1p(exp(x));
opt  = Descent();
Q    = 16;
ns   = 2;
norm = DefEcon.mynorm;
vf   = EconSim.Sim[:, 6];
st   = EconSim.Sim[:, 2:3];
defstatus = EconSim.Sim[:, 5];

vnd = vf[defstatus .== 0]
snd = st[defstatus .== 0, :]
NetWorkND = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Lnd(x, y) = Flux.mse(NetWorkND(x), y);
NseTnD = (mhat = NetWorkND, loss = Lnd, opt = opt);
VNDhat = DefEcon.NeuTra(vnd, snd, NseTnD, normi, Nepoch = 10);
graph_neural(VNDhat, "ValueNoDefault", ["VNDneural.png" "VNDNeuralSmpl.png"])


vd = vf[defstatus .== 1];
sd = st[defstatus .== 1, :];
NetWorkD = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Ld(x, y) = Flux.mse(NetWorkD(x), y);
NseTD = (mhat = NetWorkD, loss = Ld, opt = opt);
VDhat = DefEcon.NeuTra(vd, sd, NseTD, normi, Nepoch = 10);
graph_neural(VDhat, "ValueDefault", ["VDneural.png" "VDNeuralSmpl.png"])
