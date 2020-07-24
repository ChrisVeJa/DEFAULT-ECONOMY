###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,
   Distributions, Statistics, LinearAlgebra, Plots, StatsBase,
   Parameters, Flux;
include("DefEcon.jl");

############################################################
# []  Functions
############################################################
############################################################
#           [] SETTING - SOLVING - SIMULATING
############################################################
tsim  = 100000;
tburn = 0.05;
params, hdef, uf = DefEcon.ModelSettings();
#=
 Parameters of the model
=#
params = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);
#=
 Solving
=#
econdef = DefEcon.Solver(params, hdef, uf);
polfun = econdef.PolFun;
ext = econdef.ext;


updated!(VR, VD, V,uvr, uvd,params,ext) = begin
   VR1 = uvr + params.β * ( V * ext.P')
   VD1 = uvd' .+ params.β * (params.θ*(V * ext.P')+ (1-params.θ)*VD*ext.P')
   V1 , D1  = (max.(VD1,VR1), 1 .* (VD1 .> VR1))
   dif = maximum(abs.(V1-V))
   return VR1, VD1, D1, V1, dif
end
set = (VR, VD, V)

fixedp(params, polfun, ext) = begin
   VD  = zeros(params.ne,params.nx)
   VR  = zeros(params.ne,params.nx)
   V , D  = (max.(VD,VR), 1 .* (VD .> VR))
   qbp = polfun.q .* polfun.bp;
   cc  = ext.bgrid .+ ext.ygrid' - qbp;
   cc[cc.<0] .= 0;
   uvr = uf.(cc,params.σrisk)
   uvd = uf.(ext.ydef,params.σrisk)
   dif = 1
   iteration = 1
   while dif > 0.01 & iteration<1000
      VR, VD, D, V, dif = updated!(VR, VD, V,uvr, uvd,params,ext)
      iteration+=1
   end
   return VR, VD, D, V, dif
end

#=
 Simulating
=#
econsim = DefEcon.ModelSim(params,polfun, ext, nsim = tsim, burn = tburn);
DefEcon.graph_simul(econsim.sim);
pdef = round(100 * sum(econsim.sim[:, 5])/ tsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
