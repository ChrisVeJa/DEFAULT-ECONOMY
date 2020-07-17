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
tburn   = 0.05;
EconSim = DefEcon.ModelSim(Params, EconDef.PolFun, EconDef.Ext, nsim = tsim, burn = tburn);
NDef    = sum(EconSim.Sim[:,5]);
PDef    = round(100*NDef/ tsim; digits = 2);
display("Simulation finished, with $NDef defaults event and a frequency of $PDef")


#= +++++++++++++++++++++++++++++++++
 Training the neural network
+++++++++++++++++++++++++++++++++ =#
# [DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) j]
vf = EconSim.Sim[:, 6];
st = EconSim.Sim[:, 2:3];
defstatus = EconSim.Sim[:, 5];
vnd = vf[defstatus .== 0]
snd = st[defstatus .== 0, :];
vd = vf[defstatus .== 1]
sd = st[defstatus .== 1, :];
ϕf(x)= log1p(exp(x)) ;
Q  = 16; ns = 2;


myrun(typ,v,s) = begin
    anim = @animate for rolh in 1:20
        if typ == 1
            v1 = v; s1 = s;
        elseif typ ==2
            v1 = vd; s1 = DefEcon.mynorm(s);
        else
            v1 = DefEcon.mynorm(v);
            s1 = DefEcon.mynorm(s);
        end
        NetWork = Chain(Dense(ns, Q, ϕf), Dense(Q, 1));
        Ld(x, y) = Flux.mse(NetWorkD(x), y);
        dataD = Flux.Data.DataLoader(s1', v1')
        ps = Flux.params(NetWork)
        Flux.@epochs 10 Flux.Optimise.train!(Ld, ps, dataD, Descent());
        v1hat = NetWork(s1');
        v1hat = convert(Array{Float64}, v1hat);
        dplot = [v1 v1hat'];
        plot(dplot[1:500,:], legend= :topleft, label =["actual" "hat"],
            fg_legend=:transparent, legendfontsize = 6, c =[:blue :red],
            w = [0.75 0.5], style= [:solid :dash])
    end
    return anim
end

anim1 = myanim(1,vnd,snd);
anim2 = myanim(2,vnd,snd);
anim3 = myanim(3,vnd,snd);
anim4 = myanim(1,vd,sd);
anim5 = myanim(2,vd,sd);
anim6 = myanim(3,vd,sd);



gif(anim1, "gif1.gif", fps = 5);
gif(anim2, "gif2.gif", fps = 5);
gif(anim3, "gif3.gif", fps = 5);




# This gives us a reason why we need to use both f(x), x normalized
PolFun1,NeuralNDold, NeuralDoldf = try1(Params, EconDef);
PolFunEnd,NeuralNDEnd, NeuralDEnd = try2(PolFun1,NeuralNDold, NeuralDoldf, Params, EconDef.Ext, uf, tsim, tburn)
