###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Distributions, Statistics, LinearAlgebra, StatsBase
using Parameters, Flux, ColorSchemes, Gadfly
using Cairo, Fontconfig, Tables, DataFrames, Compose
include("supcodes.jl");

############################################################
# [1]  FUNCTIONS TO BE USED
############################################################
# ----------------------------------------------------------
# [1.a]  Training of neural network
# ----------------------------------------------------------
mytrain(NN, data) = begin
    lossf(x, y) = Flux.mse(NN(x), y)
    pstrain = Flux.params(NN)
    Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, data, Descent())
end
# ----------------------------------------------------------
# [1.b]  Policy function conditional on expected values
# ----------------------------------------------------------
update_solve(vr, vd, settings, params, uf,hf) = begin
    @unpack P, b, y = settings;
    @unpack r,β, θ, σrisk, nx, ne, fhat = params;
    p0 = findmin(abs.(0 .- b))[2]
    udef = uf(hf(y, fhat), σrisk);
    udef = repeat(udef', ne, 1);
    yb = b .+ y';
    # ----------------------------------------
    vf = max.(vr,vd);
    D  = 1 * (vd .> vr)
    bb   = repeat(b, 1, nx)
    vf1, vr1, vd1, D1, bp, q, eδD =
        value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf);
    return (vf = vf1, vr = vr1, vd = vd1, D = D1,
            bb = bb[bp], q = q, eδD = eδD)
end

# ----------------------------------------------------------
# [1.c] Convergence of neural networks
# ----------------------------------------------------------
mydata(sim) = begin
    x  = sim[:,[2,4,9]];
    lx = minimum(x, dims = 1);
    ux = maximum(x, dims = 1);
    xst = 2 * (x .- lx) ./ (ux - lx) .- 1;
    return xst,(lx,ux);
end
function updNN(NN_a,polfun,opts)
    @unpack sswho, set, par, uf, hf, Nsim = opts
    β0, myNN = Flux.destructure(NN_a);
    simaux = ModelSim(par, polfun, set, hf, nsim = Nsim);
    xst1,(lx1,ux1) = mydata(simaux.sim);
    (sst1,vrt1) = (xst1[:, 1:2], xst1[:, 3]);
    sswhole =  2 * (sswho .- lx1[1:2]') ./ (ux1[1:2]' - lx1[1:2]') .- 1;
    loss(x, y) = Flux.mse(NN_a(x), y)
    ps = Flux.Params(Flux.params(NN_a));
    gs = gradient(ps) do
      loss(sst',vrt');
    end
    Flux.update!(Descent(),ps,gs);
    β1,  = Flux.destructure(NN_a);
    difNN= maximum(abs.(β0-β1));
    newβ = 0.9*β1 + 0.1*β0 ;
    NN_a = myNN(newβ);
    vrhat = ((1 / 2 * (NN_a(sswhole')' .+ 1)) * (ux1[3] - lx1[3]) .+ lx1[3])
    vrhat = reshape(vrhat, par.ne, par.nx)
    polfunfit = update_solve(vrhat, polfun.vd, set, par, uf,hf);
    return NN_a, polfunfit,difNN;
end
function convergenNN!(NN,polfun, opts)
    difNN = 1;
    iter = 1
    while difNN>3e-6 && iter<2000
        NN_a, polfunfit,difNN = updNN(NN,polfun,opts);
        NN = NN_a;
        polfun = polfunfit;
        iter+=1;
        if mod(iter,20)==0
            print("Maximum distance in iteration $iter: $difNN \n")
        end
    end
    return NN, polfun;
end
# ------------------------------------------------------------------
# [1.d] Functions for plotting
# ------------------------------------------------------------------
_myplot(ModelData,x, titlex) = Gadfly.plot(ModelData, x = "debt", y = x,
    color = "output", Geom.line, Guide.ylabel(""),
    Theme(background_color = "white",key_position = :right,
    key_title_font_size = 6pt, key_label_font_size = 6pt),
    Guide.xlabel("Debt (t)"), Guide.title(titlex));

_myheatD(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
    y = "output", color = "D",Geom.rectbin, Theme(background_color = "white",
    key_title_font_size = 8pt, key_label_font_size = 8pt),
    Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
    Guide.yticks(ticks = yticks), Scale.color_discrete_manual("red","green"),
    Theme(background_color = "white"),  Guide.colorkey(title = "Default choice",
    labels = ["Default","No Default"]),
    Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

_myheatDS(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
        y = "output", color = "D",Geom.rectbin, Theme(background_color = "white",
        key_title_font_size = 8pt, key_label_font_size = 8pt),
        Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
        Guide.yticks(ticks = yticks), Scale.color_discrete_manual("green","red","white"),
        Theme(background_color = "white"),  Guide.colorkey(title = "Default choice",
        labels = ["Default","No Default",""]),
        Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

_myheatPD(ModelData,yticks) = Gadfly.plot(ModelData, x = "debt",
        y = "output", color = "pd",Geom.rectbin, Theme(background_color = "white",
        key_title_font_size = 8pt, key_label_font_size = 8pt),
        Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
        Guide.yticks(ticks = yticks), Theme(background_color = "white"),
        Scale.color_continuous(colormap=Scale.lab_gradient("midnightblue",
        "white", "yellow1")),
        Guide.colorkey(title = "Probability of Default"),
        Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));

myplot(xs,set,par; simulation=false) = begin
    if ~simulation
        MoDel = hcat([vec(i) for i in xs]...);
        MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) MoDel];
        heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q, :pd];
        ModelData = DataFrame(Tables.table(MoDel, header = heads));
    else
        mataux = unique(xs.sim[:,2:end],dims=1); # only uniques
        heads = [:debt, :output, :yj, :b, :bj,:D,:vf, :vr, :vd, :q, :pd,:θ];
        MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))]
        MatAux= [MoDel fill(NaN,size(MoDel,1),length(heads)-2)]; #whole grid
        MatAux[:,end-1].=0.5 ; # just for plotting
        for i in 1:size(mataux,1)
            l1 = findfirst(x -> x == mataux[i, 1], set.b)
            l2 = findfirst(x -> x == mataux[i, 3], set.y)
            MatAux[(l2-1)*par.ne+l1,3:end] = mataux[i,3:end]
        end
        ModelData = DataFrame(Tables.table(MatAux, header = heads));
        sort!(ModelData, :D);
    end
    yticks = round.(set.y, digits = 2);
    yticks = [yticks[1], yticks[6], yticks[11], yticks[16], yticks[end]];
    vars = ["vf" "vr" "vd" "b"]
    tvars =
        ["Value function" "Value of repayment" "Value of default" "PF for debt"]
    ppplot = Array{Any}(undef,6)
    for i = 1:length(vars)
        ppplot[i] = _myplot(ModelData,vars[i], tvars[i])
    end
    if ~simulation
        ppplot[5] = _myheatD(ModelData,yticks)
    else
        ppplot[5] = _myheatDS(ModelData,yticks)
    end
    ppplot[6] = _myheatPD(ModelData,yticks)
    return ppplot;
end

############################################################
# [2] SETTING
############################################################
par = ( r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282,nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 1001);
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk);
hf(y, fhat) = min.(y, fhat * mean(y));


############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
@time polfun, set = Solver(par, hf, uf);
Nsim = 100000
sim0 = ModelSim(par, polfun, set, hf, nsim = Nsim);
pdef = round(100 * sum(sim0.sim[:, 7]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
# ---------------------------------------------------
# [3.c] Simulating data from  the model
# ----------------------------------------------------------
xst,(lx,ux) = mydata(sim0.sim);
(sst,vrt) = (xst[:, 1:2], xst[:, 3])
traindata = Flux.Data.DataLoader((xst[:, 1:2]', xst[:, 3]'));
sswho = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))];
# Estimation
d = 16
opts = (sswho=sswho,set=set,par=par,uf=uf,hf=hf, Nsim= Nsim)

NN1 = Chain(Dense(2, d, softplus), Dense(d, 1));
NN2 = Chain(Dense(2, d, tanh), Dense(d, 1));
NN3 = Chain(Dense(2, d, elu), Dense(d, 1));
NN4 = Chain(Dense(2, d, sigmoid), Dense(d, 1));
NN5 = Chain(Dense(2, d, swish), Dense(d, 1));

models = (NN1,NN2,NN3,NN4,NN5);
Results = Array{Any,2}(undef,length(models),5)
j = 1;
for nn in models
    _nn = nn;
    mytrain(_nn,traindata)
    _nn, _polfun = convergenNN!(_nn,polfun, opts);
    _sims = ModelSim(par, _polfun, set, hf, nsim = Nsim);
    _plots = myplot(_sims,set,par, simulation=true);
    Results[j,:] = [string(_nn), _nn, _polfun, _sims, _plots];
    j += 1;
end



















mytrain(NN1,traindata);
mytrain(NN2,traindata);
mytrain(NN3,traindata);
mytrain(NN4,traindata);
mytrain(NN5,traindata);

NN1, polfun1 = convergenNN!(NN1,polfun, opts);
NN2, polfun2 = convergenNN!(NN2,polfun, opts);
NN3, polfun3 = convergenNN!(NN3,polfun, opts);
NN4, polfun4 = convergenNN!(NN4,polfun, opts);
NN5, polfun5 = convergenNN!(NN5,polfun, opts);

sim_1 = ModelSim(par, polfun1, set, hf, nsim = Nsim);
sim_2 = ModelSim(par, polfun2, set, hf, nsim = Nsim);
sim_3 = ModelSim(par, polfun3, set, hf, nsim = Nsim);
sim_4 = ModelSim(par, polfun4, set, hf, nsim = Nsim);
sim_5 = ModelSim(par, polfun5, set, hf, nsim = Nsim);

plotS1 = myplot(sim_1,set,par, simulation=true);
plotS2 = myplot(sim_2,set,par, simulation=true);
plotS3 = myplot(sim_3,set,par, simulation=true);
plotS4 = myplot(sim_4,set,par, simulation=true);
plotS5 = myplot(sim_5,set,par, simulation=true);
