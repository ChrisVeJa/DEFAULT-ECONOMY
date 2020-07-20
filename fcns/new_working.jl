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
normi(x) = begin
   xmax = maximum(x, dims = 1)
   xmin = minimum(x, dims = 1)
   xnor = (x .- 0.5 * (xmax + xmin)) ./ (0.5 * (xmax - xmin))
   return xnor, xmax, xmin
end
simtoneu(econsim,normi) = begin
   #+++++++++++++++++++++++++++++++++
   #  Data + normalization
   dst = econsim.Sim[:, 5]
   vf = econsim.Sim[:, 6]

   # Value functions
   vnd = vf[dst.==0]
   vd  = vf[dst.==1]
   vnd, vndmax, vndmin = normi(vnd)
   vd, vdmax, vdmin = normi(vd)
   # states
   st = econsim.Sim[:, [2,8]]
   s, smax, smin = normi(st)
   snd = s[dst.==0, :]
   sd  = s[dst.==1, :][:,2];
   return (nddata=(vnd,snd),ddata =(vd,sd), limt=((vndmax,vndmin),(vdmax,vdmin),(smax,smin)))
end
training(data; nepoch = 10) = begin
   ϕf(x) = log1p(exp(x)); Q1 = 16;  Q2 = 3;
   @unpack nddata, ddata, limt = data
   vnd, snd = (nddata[1] , nddata[2])
   vd, sd = (ddata[1] , ddata[2])
   length(size(snd)) != 1 ? ns1 = size(snd)[2] : ns1=1
   length(size(sd)) != 1 ?  ns2 = size(sd)[2]  : ns2=1
   #+++++++++++++++++++++++++++++++++
   #  Neural Network for No default
   NetWorkND = Chain(Dense(ns1, Q1, ϕf), Dense(Q1, 1))
   lnd(x, y) = Flux.mse(NetWorkND(x), y)
   datand = Flux.Data.DataLoader(snd', vnd')
   psnd = Flux.params(NetWorkND)
   Flux.@epochs nepoch begin
      Flux.Optimise.train!(lnd, psnd, datand, Descent())
      display(lnd(snd', vnd'))
   end
   #+++++++++++++++++++++++++++++++++
   #  Neural Network for Default
   NetWorkD = Chain(Dense(ns2, Q2, ϕf), Dense(Q2, 1))
   ld(x, y) = Flux.mse(NetWorkD(x), y)
   datad = Flux.Data.DataLoader(sd', vd')
   psd = Flux.params(NetWorkD)
   Flux.@epochs nepoch begin
      Flux.Optimise.train!(ld, psd, datad, Descent())
      display(ld(sd', vd'))
   end
   return NetWorkND, NetWorkD;
end
unpack(params, ext, uf) = begin
   @unpack r, β, θ, σrisk = params
   @unpack bgrid, ygrid, ydef, P = ext
   nx = length(ygrid)
   ne = length(bgrid)
   yb = bgrid .+ ygrid'
   BB = repeat(bgrid, 1, nx)
   p0 = findmin(abs.(0 .- bgrid))[2]
   states = [repeat(bgrid, nx, 1) repeat(ygrid, inner = (ne, 1))]
   #stateD = [repeat(bgrid, nx, 1) repeat(ydef, inner = (ne, 1))]
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
      states = states,
      utf = uf,
   )
   return mytup
end
neutopol(NetWorkND, NetWorkD, pm, neudata) = begin
   #+++++++++++++++++++++++++++++++++
   # State normalization
   smax, smin = neudata.limt[3]
   vndmax, vndmin = neudata.limt[1]
   vdmax, vdmin = neudata.limt[2]
   states = pm.states
   statend = (states .- 0.5 * (smax + smin)) ./ (0.5 * (smax - smin))
   stated = statend[:, 2]
   #+++++++++++++++++++++++++++++++++
   # Expected Values
   # [Value under no default]
   vc = NetWorkND(statend')
   vc = (0.5 * (vndmax - vndmin) * vc) .+ 0.5 * (vndmax + vndmin)
   VC = reshape(vc, pm.ne, pm.nx)
   EVC = VC * pm.P'  # Expected value of no defaulting
   # [Value under default]
   vd = NetWorkD(stated')
   vd = (0.5 * (vdmax - vdmin) * vd) .+ 0.5 * (vdmax + vdmin)
   VD = reshape(vd, pm.ne, pm.nx)
   EVD = VD * pm.P' # expected value of being in default in the next period
   #+++++++++++++++++++++++++++++++++
   # Expected price
   D = 1 * (VD .> VC)
   q = (1 / (1 + pm.r)) * (1 .- (D * pm.P'))
   qB = q .* pm.bgrid
   #+++++++++++++++++++++++++++++++++
   # UPDATING!
   # Maximization
   βEV = pm.β * EVC
   VCnew = Array{Float64,2}(undef, pm.ne, pm.nx)
   Bindex = Array{CartesianIndex{2},2}(undef, pm.ne, pm.nx)
   @inbounds for i = 1:pm.ne
      cc = pm.yb[i, :]' .- qB;
      cc[cc.<0] .= 0;
      aux_u = pm.utf.(cc, pm.σrisk) + βEV
      VCnew[i, :], Bindex[i, :] = findmax(aux_u, dims = 1)
   end
   #+++++++++++++++++++++++++++++++++
   # Value of default
   βθEVC0 = pm.β * pm.θ * EVC[pm.p0, :]
   VDnew = βθEVC0' .+ (pm.udef' .+ pm.β * (1 - pm.θ) * EVD)
   #+++++++++++++++++++++++++++++++++
   # Optimal choice
   VFnew = max.(VCnew, VDnew)
   Dnew = 1 * (VDnew .> VCnew)
   qnew = (1 / (1 + pm.r)) * (1 .- (Dnew * pm.P'))
   BPnew = pm.BB[Bindex]
   #BPnew = (1 .- Dnew) .* BPnew
   polfunnew =
   (VF = VFnew, VC = VCnew, VD = VDnew, D = Dnew, BP = BPnew, Price = qnew, Bindex = Bindex)
   return polfunnew;
end

updateneu!(NetWork1,NetWork2,data) = begin
   #+++++++++++++++++++++++++++++++++
   set1old = Flux.destructure(NetWork1); # Structure 1
   set2old = Flux.destructure(NetWork2); # Structure 2
   #+++++++++++++++++++++++++++++++++
   @unpack nddata, ddata, limt = data;
   vnd1, snd1 = (nddata[1] , nddata[2])
   vd1, sd1 = (ddata[1] , ddata[2])
   length(size(snd1)) != 1 ? ns1 = size(snd1)[2] : ns1=1;
   length(size(sd1))  != 1 ?  ns2 = size(sd1)[2]  : ns2=1;
   #+++++++++++++++++++++++++++++++++
   lnd(x, y) = Flux.mse(NetWork1(x), y)
   datand    = (snd1', vnd1');
   psnd      = Flux.params(NetWork1)
   gsnd = gradient(psnd) do
      lnd(datand...)
   end
   Flux.Optimise.update!(Descent(), psnd, gsnd) # Updating 1
   ld(x, y) = Flux.mse(NetWork2(x), y)
   datad    = (sd1', vd1');
   psd      = Flux.params(NetWork2)
   gsd = gradient(psd) do
      ld(datad...)
   end
   Flux.Optimise.update!(Descent(), psd, gsd)   # Updating 2
   #+++++++++++++++++++++++++++++++++
   # Updating
   set1new = Flux.destructure(NetWork1); # Structure 1
   set2new = Flux.destructure(NetWork2); # Structure 2
   #ψ1 = 0.8 * set1old[1][:] + 0.2 * set1new[1][:];
   #ψ2 = 0.8 * set2old[1][:] + 0.2 * set2new[1][:];
   # --------------------------------
   # Displaying
   d1 = maximum(abs.(set1new[1][:] - set1old[1][:]));
   d2 = maximum(abs.(set2new[1][:] - set2old[1][:]));
   dif = max(d1,d2);
   display("The difference is $dif")
   # --------------------------------
   #NetWork1 = set1new[2](ψ1);
   #NetWork2 = set2new[2](ψ2);
   return NetWork1, NetWork2;
end
############################################################
#           [] SETTING - SOLVING - SIMULATING
############################################################
tsim = 100000;
tburn   = 0.05;
params, hdef, uf = DefEcon.ModelSettings();
#=
 Parameters of the model
=#
params = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
    θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
    ub = 0.0, lb = -0.4, tol = 1e-8, maxite = 1e3,
);
#=
 Solving
=#
econdef = DefEcon.SolveR(params, hdef, uf);
polfun = econdef.PolFun;
ext = econdef.Ext;
#=
 Simulating
=#
econsim = DefEcon.ModelSim(params,polfun, ext, nsim = tsim, burn = tburn);
#=
 Some important messages
=#
ndef = sum(econsim.Sim[:, 5]);
pdef = round(100 * ndef / tsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events")
jj = econsim.Sim[:,8][econsim.Sim[:,5] .== 1];
jj = length(unique(jj));
display("Total # of y point $jj");

############################################################
# Training the neural network
############################################################
# [DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) j]
neudata = simtoneu(econsim,normi);
NetWorkND, NetWorkD = training(neudata);
vdhat = NetWorkD(neudata.ddata[2]');
vdhat = convert(Array{Float64}, vdhat);
dplot = [neudata.ddata[1] vdhat'];
plot(
   dplot, legend = :topleft, label = ["actual" "hat"],
   fg_legend = :transparent, legendfontsize = 6, c = [:blue :red],
   w = [0.75 0.5], style = [:solid :dash],
   title = "Value function under Default", titlefontsize = 10,
)
############################################################
# Solving the model giving the set of parameters
############################################################
pm = unpack(params, ext, uf)
polfunN = neutopol(NetWorkND, NetWorkD, pm, neudata)
econsimN = DefEcon.ModelSim(params, polfunN, ext, nsim= tsim, burn= tburn)
ndef = sum(econsimN.Sim[:,5])
pdef = round(100*ndef/ tsim; digits = 2)
display("Simulation finished, with a frequency of $pdef % of default events")
############################################################
# updating
############################################################
neudataN = simtoneu(econsimN,normi)
NetWorkND,NetWorkD = updateneu!(NetWorkND,NetWorkD,neudataN)
polfunN1 = neutopol(NetWorkND, NetWorkD, pm, neudataN)
econsimN1 = DefEcon.ModelSim(params, polfunN1, ext, nsim= tsim, burn= tburn)
ndef = sum(econsimN1.Sim[:,5])
pdef = round(100*ndef/ tsim; digits = 2)
display("Simulation finished, with a frequency of $pdef % of default events")
jj = econsimN.Sim[:,8][econsimN1.Sim[:,5] .== 1];
jj = length(unique(jj));
display("Total # of unique def events are $jj");
