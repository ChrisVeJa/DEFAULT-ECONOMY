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
simtoneu1(econsim,normi) = begin
   #+++++++++++++++++++++++++++++++++
   #  Data + normalization
   dst = econsim.sim[:, 5]
   vf = econsim.sim[:, 6]

   # Value functions
   vr = vf[dst.==0]
   vd  = vf[dst.==1]
   vr, vrmax, vrmin = normi(vr)
   vd, vdmax, vdmin = normi(vd)
   # states
   st = econsim.sim[:, [2,8]]
   s, smax, smin = normi(st)
   sr = s[dst.==0, :]
   sd = s[dst.==1, :][:,2];
   return (rdata=(vr,sr),ddata =(vd,sd), limt=((vrmax,vrmin),(vdmax,vdmin),(smax,smin)))
end
simtoneu(econsim,normi) = begin
   #+++++++++++++++++++++++++++++++++
   #  Data + normalization
   dst = econsim.sim[:, 5]
   vf  = econsim.sim[:, 6]

   # Value functions
   vf, vmax, vmin = normi(vf)
   # value function
   vr  = vf[dst.==0]
   vd  = vf[dst.==1]
   # states
   st = econsim.sim[:, [2,8]]
   s, smax, smin = normi(st)
   sr = s[dst.==0, :]
   sd = s[dst.==1, :][:,2];
   return (rdata=(vr,sr),ddata =(vd,sd), limt=((vmax,vmin),(vmax,vmin),(smax,smin)))
end
training(data; nepoch = 10) = begin
   ϕf(x) = log1p(exp(x)); Q1 = 16;  Q2 = 3;
   @unpack rdata, ddata, limt = data
   vr, sr = (rdata[1] , rdata[2])
   vd, sd = (ddata[1] , ddata[2])
   length(size(sr)) != 1 ? ns1 = size(sr)[2] :   ns1=1
   length(size(sd)) != 1 ?  ns2 = size(sd)[2]  : ns2=1
   #+++++++++++++++++++++++++++++++++
   #  Neural Network for No default
   NetWorkR = Chain(Dense(ns1, Q1, ϕf), Dense(Q1, 1))
   lr(x, y) = Flux.mse(NetWorkR(x), y)
   datar = Flux.Data.DataLoader(sr', vr')
   psr   = Flux.params(NetWorkR)
   Flux.@epochs nepoch begin
      Flux.Optimise.train!(lr, psr, datar, Descent())
      display(lr(sr', vr'))
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
   return NetWorkR, NetWorkD;
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
neutopol(NetWorkR, NetWorkD, pm, neudata) = begin
   #+++++++++++++++++++++++++++++++++
   # State normalization
   smax, smin   = neudata.limt[3]
   vrmax, vrmin = neudata.limt[1]
   vdmax, vdmin = neudata.limt[2]
   states = pm.states
   stater = (states .- 0.5 * (smax + smin)) ./ (0.5 * (smax - smin))
   stated = stater[:, 2]
   #+++++++++++++++++++++++++++++++++
   # Expected Values
   # [Value under no default]
   vr = NetWorkR(stater')
   vr = (0.5 * (vrmax - vrmin) * vr) .+ 0.5 * (vrmax + vrmin)
   vr = reshape(vr, pm.ne, pm.nx)

   # [Value under default]
   vd = NetWorkD(stated')
   vd = (0.5 * (vdmax - vdmin) * vd) .+ 0.5 * (vdmax + vdmin)
   vd = reshape(vd, pm.ne, pm.nx)

   vf = max.(vr,vd)
   D = 1 * (vd .> vr)
   evf = vf*pm.P';
   evd = vd * pm.P' # expected value of being in default in the next period
   #+++++++++++++++++++++++++++++++++
   # Expected price
   q = (1 / (1 + pm.r)) * (1 .- (D * pm.P'))
   qB = q .* pm.bgrid
   #+++++++++++++++++++++++++++++++++
   # UPDATING!
   # Maximization
   βevf = pm.β * evf
   vrnew = Array{Float64,2}(undef, pm.ne, pm.nx)
   bindex = Array{CartesianIndex{2},2}(undef, pm.ne, pm.nx)
   @inbounds for i = 1:pm.ne
      cc = pm.yb[i, :]' .- qB;
      cc[cc.<0] .= 0;
      aux_u = pm.utf.(cc, pm.σrisk) + βevf;
      vrnew[i, :], bindex[i, :] = findmax(aux_u, dims = 1)
   end
   #+++++++++++++++++++++++++++++++++
   # Value of default
   βθevf = pm.β * pm.θ * evf[pm.p0, :]
   vdnew = βθevf' .+ (pm.udef' .+ pm.β * (1 - pm.θ) * evd)
   #+++++++++++++++++++++++++++++++++
   # Optimal choice
   vfnew = max.(vrnew, vdnew)
   Dnew = 1 * (vdnew .> vrnew)
   qnew = (1 / (1 + pm.r)) * (1 .- (Dnew * pm.P'))
   bpnew = pm.BB[bindex]
   #BPnew = (1 .- Dnew) .* BPnew
   polfunnew =
   (vf = vfnew, vr = vrnew, vd = vdnew, D = Dnew, bp = bpnew, q = qnew)
   polfunint =
   (vf = vf, vr = vr, vd = vd, D = D, q = q)
   return polfunnew, polfunint;
end
updateneu!(NetWork1,NetWork2,data) = begin
   #+++++++++++++++++++++++++++++++++
   set1old = Flux.destructure(NetWork1); # Structure 1
   set2old = Flux.destructure(NetWork2); # Structure 2
   #+++++++++++++++++++++++++++++++++
   @unpack rdata, ddata, limt = data;
   vr1, sr1 = (rdata[1] , rdata[2])
   vd1, sd1 = (ddata[1] , ddata[2])
   length(size(sr1)) != 1 ? ns1 = size(sr1)[2] : ns1=1;
   length(size(sd1)) != 1 ? ns2 = size(sd1)[2] : ns2=1;
   #+++++++++++++++++++++++++++++++++
   lr(x, y) = Flux.mse(NetWork1(x), y)
   datar    = (sr1', vr1');
   psr      = Flux.params(NetWork1)
   gsr = gradient(psr) do
      lr(datar...)
   end
   Flux.Optimise.update!(Descent(), psr, gsr) # Updating 1
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

#=
Plotting the solution
=#
DefEcon.graph_solve(polfun,ext)

#=
 Simulating
=#
econsim = DefEcon.ModelSim(params,polfun, ext, nsim = tsim, burn = tburn);
DefEcon.graph_simul(econsim.sim);
pdef = round(100 * sum(econsim.sim[:, 5])/ tsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

############################################################
# Training the neural network
############################################################
# [DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) j]
neudata = simtoneu(econsim,normi);
NetWorkR, NetWorkD = training(neudata);
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
pm = unpack(params, ext, uf);
polfunN, polintN = neutopol(NetWorkR, NetWorkD, pm, neudata);
econsimN = DefEcon.ModelSim(params, polfunN, ext, nsim= tsim, burn= tburn)
pdef = round(100*sum(econsimN.sim[:,5])/ tsim; digits = 2)
display("Simulation finished, with a frequency of $pdef % of default events")
############################################################
# updating
############################################################
neudataN = simtoneu(econsimN,normi);
NetWorkR,NetWorkD = updateneu!(NetWorkR,NetWorkD,neudataN);
polfunN1, polintN1 = neutopol(NetWorkR, NetWorkD, pm, neudataN);
econsimN1 = DefEcon.ModelSim(params, polfunN1, ext, nsim= tsim, burn= tburn)
pdef = round(100*sum(econsimN1.sim[:,5])/ tsim; digits = 2)
display("Simulation finished, with a frequency of $pdef % of default events")
econsimN = econsimN1;
anim = @animate for i in 1:pm.ne
      plot([polfun.vr[i,:] polfunN.vr[i,:] polfunN1.vr[i,:]],
      fg_legend = :transparent, legend=:bottomright,
      label=["actual" "neural network" "updated"],
      xlabel = "y-grid", c= [:blue :purple :red], w = [1.15 1.5 1.15],
      style = [:solid :dot :dash], legendtitle = "Point in grid: $i",
      legendtitlefontsize = 8)
   end every 5
gif(anim, "VR.gif", fps = 5);

anim = @animate for i in 1:pm.nx
      plot(pm.bgrid,[polfun.bp[:,i] polfunN.bp[:,i] polfunN1.bp[:,i]],
      fg_legend = :transparent, legend=:bottomright,
      label=["actual" "neural network" "updated"],
      xlabel = "y-grid", c= [:blue :purple :red], w = [1.15 1.5 1.15],
      style = [:solid :dot :dash], legendtitle = "state: $i",
      legendtitlefontsize = 8)
   end
gif(anim, "Bp.gif", fps = 1);

anim = @animate for i in 1:pm.nx
      plot(pm.bgrid,[polfun.q[:,i] polfunN.q[:,i] polfunN1.q[:,i]],
      fg_legend = :transparent, legend=:topleft,
      label=["actual" "neural network" "updated"],
      xlabel = "y-grid", c= [:blue :purple :red], w = [1.15 1.5 1.15],
      style = [:solid :dot :dash], legendtitle = "state: $i",
      legendtitlefontsize = 8)
   end
gif(anim, "q.gif", fps = 1);

anim = @animate for i in 1:pm.nx
      a2 = polfunN.D[:,i].+ 0.01
      a3 = polfunN1.D[:,i].+0.02
      scatter(pm.bgrid,[polfun.D[:,i]  a2  a3],
      fg_legend = :transparent, legend=:bottomleft, markersize = 4,
      label=["actual" "neural network" "updated"], markerstrokewidth = 0.1,
      xlabel = "y-grid", c= [:blue :purple :red], w = [1.15 1.5 1.15],
      style = [:solid :dot :dash], legendtitle = "state: $i",
      legendtitlefontsize = 6, legendfontsize= 6)
   end
gif(anim, "D.gif", fps = 1);

plot([polfun.vd[end,:] polfunN.vd[end,:] polfunN1.vd[end,:]],
      fg_legend = :transparent, legend=:bottomright,
      label=["actual" "neural network" "updated"],
      xlabel = "y-grid", c= [:blue :purple :red], w = [1.15 1.5 1.15],
      style = [:solid :dot :dash])
savefig("VD.png");
