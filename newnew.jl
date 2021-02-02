###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our basic functions
############################################################
using Random, Distributions, Statistics, LinearAlgebra, StatsBase
using Parameters, ColorSchemes, Gadfly, Flux
using Cairo, Fontconfig, Tables, DataFrames, Compose
include("supcodes.jl");

############################################################
# [1]  FUNCTIONS TO BE USED
############################################################
# ----------------------------------------------------------
# Function to update the policies function
update_solve(hat_vr, hat_vd, settings, params, uf) = begin
    # Unpacking some parameters
    @unpack P, b, y = settings;
    @unpack r, β, ne, nx, σrisk, θ = params;
    p0 = findmin(abs.(0 .- b))[2]; # position of the 0 debt
    udef = repeat(settings.udef', ne, 1) # utility function
    # Default choice
    hat_vf = max.(hat_vr, hat_vd);
    hat_D = 1 * (hat_vd .> hat_vr); # default choice
    # Expecting values
    evf1 = hat_vf * P'; # value function
    evd1 = hat_vd * P'; # value of default
    eδD1 = hat_D * P';  # probability of default
    q1 = (1 / (1 + r)) * (1 .- eδD1) # price
    # Some initial calculations
    qb1 = q1 .* b; # Debt value
    βevf1 = β * evf1;# Discounted expected value
    yb = b .+ y';
    # Pre-allocating memory
    vrnew = Array{Float64,2}(undef, ne, nx)
    cc1 = Array{Float64,2}(undef, ne, nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    # Optimal debt issuing - value of repayment
    @inbounds for i = 1:ne
        cc1 = yb[i, :]' .- qb1;
        cc1 = max.(cc1, 0);
        aux_u = uf.(cc1, σrisk) + βevf1;
        vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1);
    end
    bb = repeat(b, 1, nx);
    bb = bb[bpnew];
    # Results
    # expected value next period if you default
    evaux = θ * evf1[p0, :]' .+ (1 - θ) * evd1;
    vdnew = udef + β * evaux; #updating default value
    vfnew = max.(vrnew, vdnew);# updating value function
    Dnew = 1 * (vdnew .> vrnew); # default choice
    eδD = Dnew * P';   # Probabilty of default
    qnew = (1 / (1 + r)) * (1 .- eδD); # Price
    return (vf = vfnew, vr = vrnew, vd = vdnew,
            D = Dnew, bb = bb, q = qnew, bp = bpnew)
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
_updateNN(NN,loss,lossbls,ss,ww,yy)= begin
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
_estNN(NN,loss,ss,ww,yy, lossbls) = begin
    dif = 1;  rep = 0; NN1 = nothing;
    while dif>1e-5 && rep<5000
        NN1, dif = _updateNN(NN,loss,lossbls,ss,ww,yy);
        NN = NN1;
        rep +=1;
    end
    display("Iteration $rep: maximum dif $dif")
    return NN1;
end

# ----------------------------------------------------------
# Functions for plotting
# Heatmap for default choice
myheat(xs,set,par;simulation = 0) = begin
    if simulation == 0
        MoDel = hcat([vec(i) for i in xs]...);
        MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1)) MoDel];
        heads = [:debt, :output, :vf, :vr, :vd, :D, :b, :q, :bb];
        ModelData = DataFrame(Tables.table(MoDel, header = heads));
        yticks = round.(set.y, digits = 2);
        yticks = [yticks[1], yticks[6], yticks[11], yticks[16], yticks[end]];
        heat = Gadfly.plot(ModelData, x = "debt", y = "output", color = "D",
            Geom.rectbin, Theme(background_color = "white",
            key_title_font_size = 8pt,
            key_label_font_size = 8pt), Guide.ylabel("Output (t)"),
            Guide.xlabel("Debt (t)"), Guide.yticks(ticks = yticks),
            Scale.color_discrete_manual("red","green"),
            Theme(background_color = "white"),
            Guide.colorkey(title = "Default choice",
            labels = ["Default","No Default"]),
            Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]));
    else
        ytick = round.(set.y, digits = 2);
        yticks = [ytick[1], ytick[6], ytick[11], ytick[16], ytick[end]];
        mataux = dropdims(xs.sim[:,[2 8 5 9]], dims=2); # we pick only y_j
        mataux = unique(mataux,dims=1); # only uniques
        heads = [:debt, :output, :D, :vr];
        MoDel = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))]
        MatAux= [MoDel fill(NaN,size(MoDel,1),2)]; #whole grid
        for i in 1:size(mataux,1)
            l1 = findfirst(x -> x == mataux[i, 1], set.b)
            l2 = findfirst(x -> x == mataux[i, 2], set.y)
            MatAux[(l2-1)*par.ne+l1,3:4] = mataux[i,3:4]
        end
        MatAux = DataFrame(Tables.table(MatAux, header = heads));
        sort!(MatAux, :D);
        heat = Gadfly.plot(MatAux, x = "debt",
            y = "output", color = "D", Geom.rectbin,
            Scale.color_discrete_manual("green", "red","white"),
            Theme(background_color = "white"),
            Theme(background_color = "white",
                    key_title_font_size = 8pt, key_label_font_size = 8pt),
            Guide.ylabel("Output (t)"), Guide.xlabel("Debt (t)"),
            Guide.yticks(ticks = yticks),
            Guide.colorkey(title = "Default choice",
            labels = ["No Default", "Default"]),
            Guide.xticks(ticks = [-0.40, -0.3, -0.2, -0.1, 0]))
    end
    return heat;
end

############################################################
# [2] SETTING THE MODEL
############################################################
par = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0, fhat = 0.969, ub = 0,
        lb = -0.4, tol = 1e-8, maxite = 500, ne = 1001)
uf(x, σrisk) = x .^ (1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] THE MODEL
############################################################
# ----------------------------------------------------------
# Solving the model
polfun, set = Solver(par, hf, uf);
heat0 = myheat(polfun,set,par)

# ----------------------------------------------------------
# Simulating data from  the model
Nsim = 100_000
sim0 = ModelSim(par, polfun, set, hf, nsim = Nsim);
pdef = round(100 * sum(sim0.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
heat1 = myheat(sim0,set,par,simulation=1)

############################################################
# [4] NEURAL NETWORK
############################################################
# ---------------------------------------------------
# [4.a] The neural network with one layer
# ---------------------------------------------------
dNN=16
NN = Chain(Dense(2, dNN, softplus), Dense(dNN, 1));

# ---------------------------------------------------
# [4.b] The data
# ---------------------------------------------------
# Only data when the country is on the market
x = [sim0.sim[sim0.sim[:,end].== 0,2:3] sim0.sim[sim0.sim[:,end].== 0,9]];
lx = minimum(x,dims=1); ux = maximum(x,dims=1) ; # it is better than extrema
x0 = 2 * (x .- lx) ./ (ux - lx) .- 1; # normalization
#x0 = [x0[:,1:2] x[:,3]]; # value of repayment is not normalized
# Data for the neural network
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

# ---------------------------------------------------
# [4.c] Estimation
# ---------------------------------------------------
# I estimated 10 posible models
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
        NNopt = NN1
        lossvalue = lossaux
    end
end

# ---------------------------------------------------
# [4.d] Projection over the whole grid
# --------------------------------------------------
ssgrid = [repeat(set.b, par.nx) repeat(set.y, inner = (par.ne, 1))];
ssgrid = 2 * (ssgrid .- lx[1:2]') ./ (ux[1:2]' - lx[1:2]') .- 1;
vrhat  = (0.5 * (NNopt(ssgrid') .+ 1) .* (ux[3]-lx[3])) .+ lx[3];
vrhat  =  reshape(vrhat,par.ne,par.nx);
hat_vd = polfun.vd
polfunSfit = update_solve(vrhat, hat_vd, set, par, uf);
simfit = ModelSim(par, polfunSfit, set, hf, nsim = Nsim);
pdef1 = round(100 * sum(simfit.sim[:, 5]) / Nsim; digits = 2);
display("Simulation finished $pdef1 percent");
heat2 = myheat(polfunSfit,set,par);
heat3 = myheat(simfit,set,par,simulation=1);
