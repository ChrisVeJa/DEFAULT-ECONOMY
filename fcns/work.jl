###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,
    Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux;
include("DefEcon.jl");
include("try1.jl");
include("try2.jl");
############################################################
#[1] Setting >> Solving
############################################################
Params, hdef, uf = DefEcon.ModelSettings();
Params = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ  = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);
EconDef = DefEcon.SolveR(Params, hdef, uf);

EconSim = DefEcon.ModelSim(Params, PolFun, ExtFea, nsim = tsim, burn = tburn);
NDef    = sum(EconSim.Sim[:,5]);
PDef    = round(100*NDef/ tsim; digits = 2);
display("Simulation finished, with $NDef defaults event and a frequency of $PDef")
tsim    = 100000;
tburn   = 0.05;
PolFun, NetWorkNDold, NetWorkDold = try1(Params, EconDef, tsim = tsim, tburn = tburn);
PolFun1,NetWorkND1, NetWorkD1 = try2(PolFun,NetWorkNDold, NetWorkDold, Params, EconDef.Ext, uf, tsim, tburn)
