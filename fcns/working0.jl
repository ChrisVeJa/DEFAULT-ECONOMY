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
include("try1.jl");
include("try2.jl");
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
EconSim = DefEcon.ModelSim(Params, EconDef.PolFun, EconDef.Ext, nsim = tsim, burn = tburn);
NDef    = sum(EconSim.Sim[:,5]);
PDef    = round(100*NDef/ tsim; digits = 2);
display("Simulation finished, with $NDef defaults event and a frequency of $PDef")


#= +++++++++++++++++++++++++++++++++
 Training the neural network
+++++++++++++++++++++++++++++++++ =#
# [DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) j]
vf  = EconSim.Sim[:, 6];
st  = EconSim.Sim[:, [2,8]];
defstatus = EconSim.Sim[:, 5];
vnd = vf[defstatus .== 0]
snd = st[defstatus .== 0, :];
vd  = vf[defstatus .== 1]
sd  = st[defstatus .== 1, :];
sd  = sd[:,2];
ϕf(x)= log1p(exp(x)) ;
Q   = 16; ns = 2;


myrun(typ,v,s,Q,ns) = begin
    anim = @animate for rolh in 1:40
        if typ == 1
            v1 = v; s1 = s;
        elseif typ ==2
            v1 = v; s1 = DefEcon.mynorm(s);
        else
            v1 = DefEcon.mynorm(v);
            s1 = DefEcon.mynorm(s);
        end
        NetWork = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
        Ld(x, y) = Flux.mse(NetWork(x), y);
        dataD = Flux.Data.DataLoader(s1', v1')
        ps = Flux.params(NetWork)
        Flux.@epochs 10 Flux.Optimise.train!(Ld, ps, dataD, Descent());
        v1hat = NetWork(s1');
        v1hat = convert(Array{Float64}, v1hat);
        dplot = [v1 v1hat'];
        plot(dplot[1:200,:], legend= :topleft, label =["actual" "hat"],
            fg_legend=:transparent, legendfontsize = 6, c =[:blue :red],
            w = [0.75 0.5], style= [:solid :dash])
    end
    return anim
end

#anim1 = myrun(1,vnd,snd);
#anim2 = myrun(2,vnd,snd);
#anim3 = myrun(3,vnd,snd);
anim4 = myrun(1,vd,sd,3,1);
anim5 = myrun(2,vd,sd,3,1);
anim6 = myrun(3,vd,sd,3,1);



#gif(anim1, "gif1.gif", fps = 5);
#gif(anim2, "gif2.gif", fps = 5);
#gif(anim3, "gif3.gif", fps = 5);
gif(anim4, "gif4a.gif", fps = 5);
gif(anim5, "gif5a.gif", fps = 5);
gif(anim6, "gif6a.gif", fps = 5);

#= This gives us a reason why we need to use both f(x), x normalized
normi(x) = begin
    xmax = maximum(x)
    xmin = minimum(x)
    xnor = (x .- 0.5*(xmax+xmin)) ./ (0.5*(xmax - xmin))
    return xnor, xmax, xmin
end

# No default
vnd1, vndmax1 , vndmin1 = normi(vnd);
s1nd1, s1ndmax1 , s1ndmin1 = normi(snd[:,1]);
s2nd1, s2ndmax1 , s2ndmin1 = normi(snd[:,2]);
snd1 = [s1nd1 s2nd1];
NetWorkND = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Lnd(x, y) = Flux.mse(NetWorkND(x), y);
dataND = Flux.Data.DataLoader(snd1', vnd1')
psND = Flux.params(NetWorkND)
Flux.@epochs 10 Flux.Optimise.train!(Lnd, psND, dataND, Descent());

# No default
vd1, vdmax1 , vdmin1 = normi(vd);
s1d1, s1dmax1 , s1dmin1 = normi(sd[:,1]);
s2d1, s2dmax1 , s2dmin1 = normi(sd[:,2]);
sd1 = [s1d1 s2d1];
NetWorkD = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
Ld(x, y) = Flux.mse(NetWorkD(x), y);
dataD = Flux.Data.DataLoader(sd1', vd1')
psD = Flux.params(NetWorkD)
Flux.@epochs 10 Flux.Optimise.train!(Ld, psD, dataD, Descent());


#W Solving the model giving the set of parameters

_unpack(Params, Ext, uf) = begin
   @unpack r, β, θ, σrisk = Params
   @unpack bgrid, ygrid, ydef, P = Ext
   nx = length(ygrid)
   ne = length(bgrid)
   yb = bgrid .+ ygrid'
   BB = repeat(bgrid, 1, nx)
   p0 = findmin(abs.(0 .- bgrid))[2]
   stateND= [repeat(bgrid, nx, 1) repeat(ygrid, inner = (ne, 1))]
   stateD = [repeat(bgrid, nx, 1) repeat(ydef, inner = (ne, 1))]
   udef = uf.(ydef, σrisk)
   mytup = (
      r = r, β = β, θ = θ, σrisk = σrisk, bgrid = bgrid,
      ygrid = ygrid, nx = nx,  ne = ne,  P = P, ydef = ydef,
      udef = udef,  yb = yb, BB = BB, p0 = p0, stateND = stateND,
      stateD = stateD, utf = uf,
   )
   return mytup;
end

pm = _unpack(Params, EconDef.Ext, uf);

stateND = pm.stateND;
stateD  = pm.stateD;
stateND = (stateND .- 0.5 * ([s1ndmax1 s2ndmax1]+ [s1ndmin1 s2ndmin1])) ./ (0.5 * ([s1ndmax1 s2ndmax1] - [s1ndmin1 s2ndmin1]));
stateD  = (stateD .- 0.5 * ([s1dmax1 s2dmax1]+ [s1dmin1 s2dmin1])) ./ (0.5 * ([s1dmax1 s2dmax1] - [s1dmin1 s2dmin1]));

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
=#
