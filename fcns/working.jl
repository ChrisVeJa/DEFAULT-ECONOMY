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
#[1] Setting >> Solving
############################################################
Params, hdef, uf = DefEcon.ModelSettings();
Params = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);

EconDef = DefEcon.SolveR(Params, hdef, uf);
tsim    = 100000;
tburn   = 0.5;
EconSim = DefEcon.ModelSim(
   Params,
   EconDef.PolFun,
   EconDef.Ext,
   nsim = tsim,
   burn = tburn,
);
NDef = sum(EconSim.Sim[:, 5]);
PDef = round(100 * NDef / tsim; digits = 2);
display("Simulation finished, with frequency of $PDef default events")


############################################################
# Training the neural network
############################################################
# [DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) j]
normi(x) = begin
   xmax = maximum(x, dims = 1)
   xmin = minimum(x, dims = 1)
   xnor = (x .- 0.5 * (xmax + xmin)) ./ (0.5 * (xmax - xmin))
   return xnor, xmax, xmin
end
ϕf(x)= log1p(exp(x)) ;
Q   = 16; ns = 2;

vf = EconSim.Sim[:, 6];
st = EconSim.Sim[:, 2:3];
dst = EconSim.Sim[:, 5];
v, vmax, vmin = normi(vf);
s, smax, smin = normi(st);

vnd = v[dst.==0];
snd = s[dst.==0, :];
vd = v[dst.==1];
sd = s[dst.==1, :];

#+++++++++++++++++++++++++++++++++
#  Neural Network for No default
#+++++++++++++++++++++++++++++++++

NetWorkND = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Lnd(x, y) = Flux.mse(NetWorkND(x), y);
dataND = Flux.Data.DataLoader(snd', vnd')
psND = Flux.params(NetWorkND)
Flux.@epochs 10 begin
   Flux.Optimise.train!(Lnd, psND, dataND, Descent());
   display(Lnd(snd',vnd'));
end
vndhat = NetWorkND(snd');
vndhat = convert(Array{Float64}, vndhat);
dplot = [vnd vndhat'];
plot(
   dplot[1:250, :], legend = :topleft, label = ["actual" "hat"],
   fg_legend = :transparent, legendfontsize = 6, c = [:blue :red],
   w = [0.75 0.5], style = [:solid :dash],
   title = "Value function under No Default", titlefontsize = 10,
)
#+++++++++++++++++++++++++++++++++
#  Neural Network for Default
#+++++++++++++++++++++++++++++++++
NetWorkD = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Ld(x, y) = Flux.mse(NetWorkD(x), y);
dataD = Flux.Data.DataLoader(sd', vd')
psD = Flux.params(NetWorkD)
Flux.@epochs 10 begin
   Flux.Optimise.train!(Ld, psD, dataD, Descent());
   display(Ld(sd',vd'));
end
vdhat = NetWorkD(sd');
vdhat = convert(Array{Float64}, vdhat);
dplot = [vd vdhat'];
plot(
   dplot[1:250, :], legend = :topleft, label = ["actual" "hat"],
   fg_legend = :transparent, legendfontsize = 6, c = [:blue :red],
   w = [0.75 0.5], style = [:solid :dash],
   title = "Value function under Default", titlefontsize = 10,
)

############################################################
# Solving the model giving the set of parameters
############################################################


unpack(Params, Ext, uf) = begin
   @unpack r, β, θ, σrisk = Params
   @unpack bgrid, ygrid, ydef, P = Ext
   nx = length(ygrid)
   ne = length(bgrid)
   yb = bgrid .+ ygrid'
   BB = repeat(bgrid, 1, nx)
   p0 = findmin(abs.(0 .- bgrid))[2]
   stateND = [repeat(bgrid, nx, 1) repeat(ygrid, inner = (ne, 1))]
   stateD = [repeat(bgrid, nx, 1) repeat(ydef, inner = (ne, 1))]
   udef = uf.(ydef, σrisk)
   mytup = (
      r = r,
      β = β,
      θ = θ,
      σrisk = σrisk,
      bgrid = bgrid,
      ygrid = ygrid,
      nx = nx,
      ne = ne,
      P = P,
      ydef = ydef,
      udef = udef,
      yb = yb,
      BB = BB,
      p0 = p0,
      stateND = stateND,
      stateD = stateD,
      utf = uf,
   )
   return mytup
end

pm = unpack(Params, EconDef.Ext, uf);

stateND = pm.stateND;
stateD  = pm.stateD;
#stateND = (stateND .- 0.5 * ([s1ndmax1 s2ndmax1]+ [s1ndmin1 s2ndmin1])) ./ (0.5 * ([s1ndmax1 s2ndmax1] - [s1ndmin1 s2ndmin1]));
#stateD  = (stateD .- 0.5 * ([s1dmax1 s2dmax1]+ [s1dmin1 s2dmin1])) ./ (0.5 * ([s1dmax1 s2dmax1] - [s1dmin1 s2dmin1]));

vc = NetWorkND(stateND');
vc = (0.5 * (vndmax1- vndmin1) * vc) .+ 0.5 * (vndmax1 + vndmin1);
VC = reshape(vc, pm.ne, pm.nx);
EVC = VC * pm.P';  # Expected value of no defaulting

vd = NetWorkD(stateD');
vd = (0.5 * (vdmax1- vdmin1) * vd) .+ 0.5 * (vdmax1 + vdmin1);
VD = reshape(vd, pm.ne, pm.nx);
EVD = VD * pm.P'; # expected value of being in default in the next period

# [1.6] Income by issuing bonds
#q  = EconSol.Sol.BPrice;
q  = EconDef.PolFun.Price;
qB = q .* pm.bgrid;

# [1.7] Policy function for bonds under continuation
βEV = pm.β * EVC;
VC1 = Array{Float64,2}(undef, pm.ne, pm.nx);
Bindex = Array{CartesianIndex{2},2}(undef, pm.ne, pm.nx);
@inbounds for i in 1:pm.ne
   cc = pm.yb[i, :]' .- qB;
   cc[cc.<0] .= 0;
   aux_u = pm.utf.(cc, pm.σrisk) + βEV;
   VC1[i, :], Bindex[i, :] = findmax(aux_u, dims = 1);
end
# [1.8] Value function of default
βθEVC0 = pm.β * pm.θ * EVC[pm.p0, :];
VD1 = βθEVC0' .+ (pm.udef' .+ pm.β * (1 - pm.θ) * EVD);
# --------------------------------------------------------------
# [1.9]. New Continuation Value, Default choice, price
VF1 = max.(VC1, VD1);
D1 = 1 * (VD1 .> VC1);
q1 = (1 / (1 + pm.r)) * (1 .- (D1 * pm.P'));
BP1 = pm.BB[Bindex];
BP1 = (1 .- D1) .* BP1;
PolFun1 = (VF = VF1, VC = VC1, VD = VD1, D = D1, BP = BP1, Price = q1);

EconSim = DefEcon.ModelSim(Params, PolFun1, EconDef.Ext, nsim = tsim, burn = tburn);
NDef    = sum(EconSim.Sim[:,5]);
PDef    = round(100*NDef/ tsim; digits = 2);
display("Simulation finished, with $NDef defaults event and a frequency of $PDef")
