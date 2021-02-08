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
# Loss functions:
#   Mean squared error
loss(x,w,y,NN) = sum(w .* ((NN(x) - y).^2))/(sum(w))
#   Loss function conditional on parameters
lossbls(x,w,y,NN, θ) = begin
    be, _N0 = Flux.destructure(NN)
    NN1 = _N0(θ)
    loss = sum(w .* ((NN1(x) - y).^2))/(sum(w))
    return loss
end

# ----------------------------------------------------------
# Function for neural networks
#   Updating the parameters of the neural networks
function _updateNN(NN,loss,lossbls,ss,ww,yy)
    ##############################################
    #   θₕ₊₁ = θₕ - αₕ∇fₓ(θₜ)
    # where the updating parameter α is choosen
    # by line searching with armijo rule such that
    # f(θₕ₊₁) ≦ f(θₜ) - 1/2 αₜ||∇fₓ(θₜ) ||²
    ##############################################
    # -----------------------------------
    # Obtaining the parameters of the NN
    ps = Flux.Params(Flux.params(NN));
    # -----------------------------------
    # Calculating the gradient
    gs = gradient(ps) do
      loss(ss',ww',yy',NN);
    end # gradient, it needs to be recalculated in each iteration
    # -----------------------------------
    # Backtracking line search
    θ, re =  Flux.destructure(NN); #parameters and structure
    loss0 = loss(ss',ww',yy',NN);  #initial loss function
    ∇θ  = vcat(vec.([gs[i] for i in ps])...); # gradient vector
    ∇f0 = sum(∇θ.^2); # square of gradient
    α   = 1; # initial value of alpha
    decay = 0.3; # decay parameter
    while lossbls(ss',ww',yy',NN, θ-α.*∇θ) > loss0 - 0.5*α*∇f0
        α *= decay; #updating α
    end
    θ1 = θ-α.*∇θ; # updating θ
    θ1 = 0.9*θ1 + 0.1*θ; # new parameter vector
    dif = maximum(abs.(θ1-θ)); # difference
    NN1 = re(θ1); # new neural network
    return NN1, dif;
end
#   Estimation of the neural network
function _estNN(NN,loss,ss,ww,yy, lossbls)
    dif = 1;  rep = 0; NN1 = nothing;
    while dif>1e-5 && rep<5000
        NN1, dif = _updateNN(NN,loss,lossbls,ss,ww,yy);
        NN = NN1;
        rep +=1;
    end
    display("Iteration $rep: maximum dif $dif")
    return NN1;
end
function myestimationNN(NN,loss,ss,ww,yy,lossbls)
    θs , more = Flux.destructure(NN)
    NN1 = nothing
    NNopt = nothing
    lossvalue = 1e10
    for j in 1:10
        θaux = -1 .+ 2*rand(length(θs)); # random draws from [-1:1]
        NNaux = more(θaux);
        NN1 = _estNN(NNaux,loss,ss,ww,yy, lossbls);
        lossaux = loss(ss',ww',yy',NN1);
        display("Loss value of model $j is $lossaux")
        # Updating the neural network if the fit is better
        if lossaux < lossvalue
            NNopt = NN1;
            lossvalue = lossaux;
        end
    end
    return NNopt;
end
# ----------------------------------------------------------
# [1.c] Convergence of neural networks
# ----------------------------------------------------------
mydata(sim0) = begin
    x = [sim0.sim[sim0.sim[:,end].== 0,2:3] sim0.sim[sim0.sim[:,end].== 0,9]];
    lx = minimum(x,dims=1); ux = maximum(x,dims=1) ; # it is better than extrema
    x0 = 2 * (x .- lx) ./ (ux - lx) .- 1; # normalization
    x0Tu = [Tuple(x0[i, :]) for i = 1:size(x0, 1)]
    x1Tu = unique(x0Tu); # unique simulated states
    xx = countmap(x0Tu); # number of repetitions (in dictionary version)
    x1 = Array{Float32,2}(undef,size(x1Tu)[1],3)
    for i in 1:size(x1Tu)[1]
        x1[i,:] = [x1Tu[i]...]'
    end
    ww  = [get(xx, i, 0) for i in x1Tu] ; # weights
    ss  = x1[:,1:2]; # states
    yy  = x1[:,3]; # value of repayment
    return (ss,yy,ww),(lx,ux);
end
function updNN(NN_a,polfun,opts)
    @unpack sswho, set, par, uf, hf, Nsim,loss,lossbls = opts
    β0, myNN = Flux.destructure(NN_a);
    simaux = ModelSim(par, polfun, set, hf, nsim = Nsim);
    (sst1,vrt1,wwt1),(lx1,ux1) = mydata(simaux);
    sswhole =  2 * (sswho .- lx1[1:2]') ./ (ux1[1:2]' - lx1[1:2]') .- 1;
    NN_a, dif = _updateNN(NN_a,loss,lossbls,sst1,wwt1,vrt1);
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
(sst,vrt,wwt),(lx,ux) = mydata(sim0);
sswho = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))];
# Estimation
d = 16
opts = (sswho=sswho,set=set,par=par,uf=uf,hf=hf, Nsim= Nsim,
        loss = loss, lossbls= lossbls)

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
    _nn = = myestimationNN(_nn,loss,sst,wwt,vrt,lossbls)
    _nn, _polfun = convergenNN!(_nn,polfun, opts);
    _sims = ModelSim(par, _polfun, set, hf, nsim = Nsim);
    _plots = myplot(_sims,set,par, simulation=true);
    Results[j,:] = [string(_nn), _nn, _polfun, _sims, _plots];
    j += 1;
end
