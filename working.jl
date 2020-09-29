###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################
using Random, Distributions, Statistics, LinearAlgebra, Plots,
    StatsBase, Parameters, Flux;
include("supcodes.jl");
############################################################
# []  Functions
############################################################

############################################################
# SETTING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 251);
uf(x, σrisk)= x.^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# Solving
############################################################
polfun, settings = Solver(params, hf, uf);
plot(settings.b, polfun.vf[:,10:12], label = ["low" "normal" "high"],
    fg_legend = :transparent, bg_legend = :transparent, legend = :topleft,
    grid = :false, c= [:red :black :blue], w = [2 1.15 1.15],
    style = [:dot :solid :dash], title = "Value function by debt level",
    xlabel = "debt" , ylabel = "Value function")
savefig("./Figures/ValueFunction.png")
plot(settings.b, polfun.bb[:,10:12], label = ["low" "normal" "high"],
    fg_legend = :transparent, bg_legend = :transparent, legend = :topleft,
    grid = :false, c= [:red :black :blue], w = [2 1.15 1.15],
    style = [:dot :solid :dash], title = "Value function by debt level",
    xlabel = "debt (state)" , ylabel = "Debt issued")
savefig("./Figures/DebtChoice.png")
plot(settings.b, polfun.q[:,10:12], label = ["low" "normal" "high"],
    fg_legend = :transparent, bg_legend = :transparent, legend = :topleft,
    grid = :false, c= [:red :black :blue], w = [2 1.15 1.15],
    style = [:dot :solid :dash], title = "New issued Bonds Price",
    xlabel = "debt (state)" , ylabel = "Price of new bonds")
savefig("./Figures/PriceBond.png")
heat = heatmap(settings.y, settings.b, polfun.D',
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt");
savefig("./Figures/heatD0.png")

############################################################
# Simulation
############################################################
econsim0 = ModelSim(params,polfun, settings,hf, nsim=100000);
pdef = round(100 * sum(econsim0.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

econsim1 = ModelSim(params,polfun, settings,hf, nsim=1000000);
pdef = round(100 * sum(econsim1.sim[:, 5])/ 1000000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");

# Plotting
plot(layout=(2,2), size = (1200,900), grid = :false)
plot!(econsim0.sim[1000:2000,3], subplot= 1, label = :false, style = :dash, ylabel = "output")
plot!(econsim0.sim[1000:2000,4], subplot= 2, label = :false, style = :dash, ylabel = "debt")
plot!(econsim0.sim[1000:2000,6], subplot= 3, label = :false, style = :dash, ylabel = "Value Function")
plot!(econsim0.sim[1000:2000,7], subplot= 4, label = :false, style = :dash, ylabel = "Price")
savefig("./Figures/simulation.png")


myunique(data) = begin
    dataTu = [Tuple(data[i,:])  for i in 1:size(data)[1]]
    dataTu = unique(dataTu)
    dataTu = [[dataTu[i]...]' for i in 1:length(dataTu)]
    data   = [vcat(dataTu...)][1]
    return data
end

data0 = myunique(econsim0.sim)
data1 = myunique(econsim1.sim)

preheatmap(data,params,settings)= begin
    sts = data[:,2:3]
    datav = data[:,5:6]
    DD  = -ones(params.ne,params.nx)
    VR  = fill(NaN,params.ne,params.nx)
    stuple = [(sts[i,1], sts[i,2])  for i in 1:size(sts)[1] ];
    for i in 1:params.ne
        bi = settings.b[i]
        for j in 1:params.nx
            yj = settings.y[j]
            pos = findfirst(x -> x==(bi,yj), stuple)
            if ~isnothing(pos)
                DD[i,j] = datav[pos, 1]
                VR[i,j] = datav[pos, 2]
            end
        end
    end
    return DD, VR
end

DD0, VR0 = preheatmap(data0,params,settings);
heat0    = heatmap(settings.y, settings.b, DD0', c = cgrad([:white, :black, :yellow]),
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
DD1, VR1 = preheatmap(data1,params,settings)
heat1    = heatmap(settings.y, settings.b, DD1', c = cgrad([:white, :black, :yellow]),
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );

# Plotting HEATMAPS by sample size
plot(heat0,heat1,layout = (2,1), size = (500,800), title = ["(Actual)" "(100 m)" "(1 millon)"])
savefig("./Figures/heatmap_D.png");

############################################################
# Training
############################################################
mynorm(x) = begin
    ux = maximum(x, dims=1)
    lx = minimum(x, dims=1)
    nx = (x .- 0.5(ux+lx)) ./ (0.5*(ux-lx))
    return nx, ux, lx
end

b   = econsim0.sim[:,2]  # bₜₜ
y   = econsim0.sim[:,2]  # yₜ
vr  = econsim0.sim[:,9]  # vᵣ
vd  = econsim0.sim[:,10] # vd
vr, uvr, lvr = mynorm(vr);
vd, uvd, lvd = mynorm(vd);
ss, uss, lss = mynorm([b y]);
ys, uby, lby = mynorm(y)

data = (Array{Float32}(ss'), Array{Float32}(vr'));
datad = (Array{Float32}(y'), Array{Float32}(vd'))

mytrain(NN,data) = begin
    lossf(x,y) = Flux.mse(NN(x),y);
    traindata  = Flux.Data.DataLoader(data)
    pstrain = Flux.params(NN)
    Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, traindata, Descent())
end


##########################################################
# NEURAL NETWORK FOR SIMULATIONS WITH SAMPLE SIZE OF 100K
##########################################################

# -------------------------------------------------------
# ESTIMATION
# -------------------------------------------------------

    # Value of Repayment
        NNR1 = Chain(Dense(2, 16, softplus), Dense(16, 1));
        mytrain(NNR1,data);
        NNR2 = Chain(Dense(2, 16, tanh), Dense(16, 1));
        mytrain(NNR2,data);
        NNR3 = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
        mytrain(NNR3,data);
        NNR4 = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));
        mytrain(NNR4,data);

    # Value of Default
        NND1 = Chain(Dense(1, 16, softplus), Dense(16, 1));
        mytrain(NND1,datad);
        NND2 = Chain(Dense(1, 16, tanh), Dense(16, 1));
        mytrain(NND2,datad);
        NND3 = Chain(Dense(1, 32, relu), Dense(32, 16,softplus), Dense(16,1));
        mytrain(NND3,datad);
        NND4 = Chain(Dense(1, 32, relu), Dense(32, 16,tanh), Dense(16,1));
        mytrain(NND4,datad);

# -------------------------------------------------------
# FORECAST ERROR
# -------------------------------------------------------

    # In-sample
        uniqdata = unique([data[1]; data[2]],dims=2)
        fitt = [NNR1(uniqdata[1:2,:])' NNR2(uniqdata[1:2,:])' NNR3(uniqdata[1:2,:])' NNR4(uniqdata[1:2,:])']
        diff = abs.(uniqdata[3,:] .- fitt)
        scatter(uniqdata[1,:],uniqdata[2,:], [diff[:,1], diff[:,2], diff[:,3], diff[:,4]],
                markerstrokewidth= 0.3, alpha =0.42,
                label =["softplus" "tanh" "relu+softplus" "relu+tanh"],
                fg_legend=:transparent, bg_legend=:transparent)
        savefig("./Figures/scatter1.png");

    # Out-sample
        bnorm = (settings.b .- 0.5(uss[1]+lss[1])) ./ (0.5*(uss[1]-lss[1]))
        ynorm = (settings.y .- 0.5(uss[2]+lss[2])) ./ (0.5*(uss[2]-lss[2]))
        vnorm = (vec(polfun.vr) .- 0.5(uvr+lvr)) ./ (0.5*(uvr-lvr))
        states = [repeat(bnorm,params.nx)' ; repeat(ynorm,inner= (params.ne,1))']
        outpre = [NNR1(states)' NNR2(states)' NNR3(states)' NNR4(states)']
        diff2 = abs.(vnorm.- outpre)
        scatter(states[1,:],states[2,:], [diff2[:,1], diff2[:,2], diff2[:,3], diff2[:,4]],
                markerstrokewidth= 0.3, alpha =0.42,
                label =["softplus" "tanh" "relu+softplus" "relu+tanh"],
                fg_legend=:transparent, bg_legend=:transparent)
        savefig("./Figures/scatter2.png");


##########################################################
# NEURAL NETWORK FOR SIMULATIONS WITH SAMPLE SIZE OF 1M
##########################################################

    b1   = econsim1.sim[:,2]  # bₜₜ
    y1   = econsim1.sim[:,2]  # yₜ
    vr1  = econsim1.sim[:,9]  # vᵣ
    vd1  = econsim1.sim[:,10] # vd
    vr1, uvr1, lvr1 = mynorm(vr1);
    vd1, uvd1, lvd1 = mynorm(vd1);
    ss1, uss1, lss1 = mynorm([b1 y1]);
    ys1, uby1, lby1 = mynorm(y1)
    data1  = (Array{Float32}(ss1'), Array{Float32}(vr1'));
    datad1 = (Array{Float32}(y1'), Array{Float32}(vd1'))

# -------------------------------------------------------
# ESTIMATION
# -------------------------------------------------------

    # Value of Repayment
    NNR1e6 = Chain(Dense(2, 16, softplus), Dense(16, 1));
    mytrain(NNR1e6,data1);
    NNR2e6 = Chain(Dense(2, 16, tanh), Dense(16, 1));
    mytrain(NNR2e6,data1);
    NNR3e6 = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
    mytrain(NNR3e6,data1);
    NNR4e6 = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));
    mytrain(NNR4e6,data1);


    # Value of Default
    NNR1e6d = Chain(Dense(1, 16, softplus), Dense(16, 1));
    mytrain(NNR1e6d,datad1);
    NNR2e6d = Chain(Dense(1, 16, tanh), Dense(16, 1));
    mytrain(NNR2e6d,datad1);
    NNR3e6d = Chain(Dense(1, 32, relu), Dense(32, 16,softplus), Dense(16,1));
    mytrain(NNR3e6d,datad1);
    NNR4e6d = Chain(Dense(1, 32, relu), Dense(32, 16,tanh), Dense(16,1));
    mytrain(NNR4e6d,datad1);

# In-sample
    uniqdata1 = unique([data1[1]; data1[2]],dims=2)
    fitte6 = [NNR1e6(uniqdata1[1:2,:])' NNR2e6(uniqdata1[1:2,:])' NNR3e6(uniqdata1[1:2,:])' NNR4e6(uniqdata1[1:2,:])']
    diffe6 = abs.(uniqdata1[3,:] .- fitte6)
    scatter(uniqdata1[1,:],uniqdata1[2,:], [diffe6[:,1], diffe6[:,2], diffe6[:,3], diffe6[:,4]],
            markerstrokewidth= 0.3, alpha =0.42,
            label =["softplus" "tanh" "relu+softplus" "relu+tanh"],
            fg_legend=:transparent, bg_legend=:transparent)
    savefig("./Figures/scatter3.png");

# Out-sample
    bnorme6 = (settings.b .- 0.5(uss1[1]+lss1[1])) ./ (0.5*(uss1[1]-lss1[1]))
    ynorme6 = (settings.y .- 0.5(uss1[2]+lss1[2])) ./ (0.5*(uss1[2]-lss1[2]))
    vnorme6 = (vec(polfun.vr) .- 0.5(uvr1+lvr1)) ./ (0.5*(uvr1-lvr1))
    statese6 = [repeat(bnorme6,params.nx)' ; repeat(ynorme6,inner= (params.ne,1))']
    outpree6 = [NNR1e6(statese6)' NNR2e6(statese6)' NNR3(statese6)' NNR4(statese6)']
    diff2e6 = abs.(vnorme6.- outpree6)

    scatter(statese6[1,:],statese6[2,:], [diff2e6[:,1], diff2e6[:,2], diff2e6[:,3], diff2e6[:,4]],
            markerstrokewidth= 0.3, alpha =0.42,
            label =["softplus" "tanh" "relu+softplus" "relu+tanh"],
            fg_legend=:transparent, bg_legend=:transparent)
    savefig("./Figures/scatter4.png");

##########################################################
# Actual
##########################################################

ss0  = [repeat(settings.b,params.nx) repeat(settings.y,inner= (params.ne,1))]   # bₜ, yₜ
vr2  = vec(polfun.vr)   # vᵣ
vr2, uvr2, lvr2 = mynorm(vr2);
ss2, uss2, lss2 = mynorm(ss0);
data2 = (Array{Float32}(ss2'), Array{Float32}(vr2'));

    NNR1act = Chain(Dense(2, 16, softplus), Dense(16, 1));
    mytrain(NNR1act,data2);
    NNR2act = Chain(Dense(2, 16, tanh), Dense(16, 1));
    mytrain(NNR2act,data2);
    NNR3act = Chain(Dense(2, 32, relu), Dense(32, 16,softplus), Dense(16,1));
    mytrain(NNR3act,data2);
    NNR4act = Chain(Dense(2, 32, relu), Dense(32, 16,tanh), Dense(16,1));
    mytrain(NNR4act,data2);

# In-sample
    fittact = [NNR1act(ss2')' NNR2act(ss2')' NNR3act(ss2')' NNR4act(ss2')']
    diffact = abs.(vr2 .- fittact )
    scatter(ss2[:,1],ss2[:,2], [diffact[:,1], diffact[:,2], diffact[:,3], diffact[:,4]],
            markerstrokewidth= 0.3, alpha =0.42,
            label =["softplus" "tanh" "relu+softplus" "relu+tanh"],
            fg_legend=:transparent, bg_legend=:transparent)
    savefig("./Figures/scatter5.png");

############################################################
#  Solving the model based on NN  results
############################################################
# Predicting with all the models
    bnorm = (settings.b .- 0.5(uss[1]+lss[1])) ./ (0.5*(uss[1]-lss[1]))
    ynorm = (settings.y .- 0.5(uss[2]+lss[2])) ./ (0.5*(uss[2]-lss[2]))
    vrnorm = (vec(polfun.vr) .- 0.5(uvr+lvr)) ./ (0.5*(uvr-lvr))
    vdnorm = (vec(polfun.vd) .- 0.5(uvd+lvd)) ./ (0.5*(uvd-lvd))
    states = [repeat(bnorm,params.nx)' ; repeat(ynorm,inner= (params.ne,1))']
    hat_vrnorm = [NNR1(states)' NNR2(states)' NNR3(states)' NNR4(states)']
    hat_vrM =  0.5*(uvr-lvr).*hat_vrnorm .+ 0.5*(uvr+lvr)
    hat_vdnorm = [NND1(ynorm')' NND2(ynorm')' NND3(ynorm')' NND4(ynorm')']
    hat_vdM =  0.5*(uvd-lvd).*hat_vdnorm .+ 0.5*(uvd+lvd)



# Starting the psolution of the model
@unpack P, b,y = settings
@unpack r, β, ne, nx, σrisk = params
hat_vr = reshape(hat_vrM[:,1],(ne,nx))
hat_vd = repeat(hat_vdM[:,1]',ne)
hat_vf = max.(hat_vr,hat_vd)
hat_D  = 1 * (hat_vd .> hat_vr)
evf1 = hat_vf * P'
eδD1 = hat_D  * P'
q1   = (1 / (1 + r)) * (1 .- eδD1) # price
qb1  = q1 .* b
βevf1= β*evf1
vrnew = Array{Float64,2}(undef,ne,nx)
cc1    = Array{Float64,2}(undef,ne,nx)
bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
yb    = b .+ y'
@inbounds for i = 1:ne
cc1 = yb[i, :]' .- qb1
cc1 = max.(cc,0)
aux_u = uf.(cc, σrisk) + βevf1
vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
end
evaux = θ * evf[p0, :]' .+  (1 - θ) * evd
vdnew = udef + β*evaux
vfnew = max.(vrnew, vdnew)
Dnew = 1 * (vdnew .> vrnew)
eδD  = Dnew  * P'
qnew = (1 / (1 + r)) * (1 .- eδD)
