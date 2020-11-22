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
# [1]  FUNCTIONS TO BE USED
############################################################
# ----------------------------------------------------------
# [1.a] Choosing unique states from the simulated data
# ----------------------------------------------------------
    myunique(data) = begin
        dataTu = [Tuple(data[i,:])  for i in 1:size(data)[1]]
        dataTu = unique(dataTu)
        dataTu = [[dataTu[i]...]' for i in 1:length(dataTu)]
        data   = [vcat(dataTu...)][1]
        return data
    end
# ----------------------------------------------------------
# [1.b] Creating the value of the grids
# ----------------------------------------------------------
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
# ----------------------------------------------------------
# [1.c]  Normalization:
#     ̃x = (x - 1/2(xₘₐₓ + xₘᵢₙ)) /(1/2(xₘₐₓ - xₘᵢₙ))
# ----------------------------------------------------------
    mynorm(x) = begin
        ux = maximum(x, dims=1)
        lx = minimum(x, dims=1)
        nx = (x .- 0.5(ux+lx)) ./ (0.5*(ux-lx))
        return nx, ux, lx
    end
# ----------------------------------------------------------
# [1.d]  Training of neural network
# ----------------------------------------------------------
    mytrain(NN,data) = begin
        lossf(x,y) = Flux.mse(NN(x),y);
        traindata  = Flux.Data.DataLoader(data)
        pstrain = Flux.params(NN)
        Flux.@epochs 10 Flux.Optimise.train!(lossf, pstrain, traindata, Descent())
    end
# ----------------------------------------------------------
# [1.d]  Policy function conditional on expected values
# ----------------------------------------------------------
    update_solve(hat_vr, hat_vd, settings,params,uf) = begin
        # Starting the psolution of the model
        @unpack P, b,y = settings
        @unpack r, β, ne, nx, σrisk,θ = params
        p0 = findmin(abs.(0 .- b))[2]
        udef = repeat(settings.udef', ne, 1)
        hat_vf = max.(hat_vr,hat_vd)
        hat_D  = 1 * (hat_vd .> hat_vr)
        evf1 = hat_vf * P'
        evd1 = hat_vd * P'
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
            cc1 = max.(cc1,0)
            aux_u = uf.(cc1, σrisk) + βevf1
            vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
        end
        bb    = repeat(b, 1, nx)
        bb    = bb[bpnew]
        evaux = θ * evf1[p0, :]' .+  (1 - θ) * evd1
        vdnew = udef + β*evaux
        vfnew = max.(vrnew, vdnew)
        Dnew  = 1 * (vdnew .> vrnew)
        eδD   = Dnew  * P'
        qnew  = (1 / (1 + r)) * (1 .- eδD)
        return (vf = vfnew, vr = vrnew, vd = vdnew, D = Dnew, bb =  bb, q = qnew, bp = bpnew)
    end

############################################################
# [2] SETTING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.9053,
        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 251);
uf(x, σrisk)= x.^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# [3] WORKING
############################################################
# ----------------------------------------------------------
# [3.a] Solving the model
# ----------------------------------------------------------
polfun, settings = Solver(params, hf, uf);
# ----------------------------------------------------------
# [3.b] "Heatmap" of Default: 1:= Default  0:= No Default
# ----------------------------------------------------------
heat = heatmap(settings.y, settings.b, polfun.D', aspect_ratio = 0.8,
    xlabel = "Output", ylabel = "Debt", legend = :false)
#savefig("./Figures/heatD0.png")

# ----------------------------------------------------------
# [3.c] Simulating data from  the model
#  Heatmap : 0 := Default
# ----------------------------------------------------------
econsim0 = ModelSim(params,polfun, settings,hf, nsim=100000);
data0 = myunique(econsim0.sim)
DD0, VR0 = preheatmap(data0,params,settings);
heat0  = heatmap(settings.y, settings.b, DD0', ylabel = "Debt",
        legend = :false, c = cgrad([:white, :black, :yellow]),
        aspect_ratio = 0.8, xlabel = "Output", )
plot(heat,heat0,layout = (2,1), size = (500,800),
     title = ["Actual" "Simulated data 100k"])
pdef = round(100 * sum(econsim0.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");
#savefig("./Figures/heatmap_D.png");

# ----------------------------------------------------------
# [3.d] Being sure that updating is working well
# ----------------------------------------------------------
hat_vr =
hat_vd =
 fafa = update_solve(hat_vr, hat_vd, settings,params,uf) = begin

############################################################
# NEURAL NETWORKS: Training
############################################################
##########################################################
# NEURAL NETWORK FOR SIMULATIONS WITH SAMPLE SIZE OF 100K
##########################################################
    b   = econsim0.sim[:,2] ; y   = econsim0.sim[:,3]  # yₜ
    vr  = econsim0.sim[:,9] ; vd  = econsim0.sim[:,10] # vd
    vr, uvr, lvr = mynorm(vr);    vd, uvd, lvd = mynorm(vd);
    ss, uss, lss = mynorm([b y]); ys, uby, lby = mynorm(y)
    data = (Array{Float32}(ss'), Array{Float32}(vr'));
    datad = (Array{Float32}(ys'), Array{Float32}(vd'))

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
        scatter(uniqdata[1,:],uniqdata[2,:], [diff[:,1], diff[:,2], diff[:,3], diff[:,4]], markerstrokewidth= 0.3, alpha =0.42,
                label =["softplus" "tanh" "relu+softplus" "relu+tanh"],legend = :topleft, fg_legend=:transparent, bg_legend=:transparent)

    # Out-sample
        bnorm = (settings.b .- 0.5(uss[1]+lss[1])) ./ (0.5*(uss[1]-lss[1]))
        ynorm = (settings.y .- 0.5(uss[2]+lss[2])) ./ (0.5*(uss[2]-lss[2]))
        vnorm = (vec(polfun.vr) .- 0.5(uvr+lvr)) ./ (0.5*(uvr-lvr))
        states = [repeat(bnorm,params.nx)' ; repeat(ynorm,inner= (params.ne,1))']
        outpre = [NNR1(states)' NNR2(states)' NNR3(states)' NNR4(states)']
        diff2 = abs.(vnorm.- outpre)
        scatter(states[1,:],states[2,:], [diff2[:,1], diff2[:,2], diff2[:,3], diff2[:,4]],markerstrokewidth= 0.3, alpha =0.42,
                label =["softplus" "tanh" "relu+softplus" "relu+tanh"], fg_legend=:transparent, bg_legend=:transparent)


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



############################################################
#  Updating based on NN
############################################################

    hat_vr = reshape(hat_vrM[:,1],(params.ne,params.nx))
    hat_vd = repeat(hat_vdM[:,1]',params.ne)
    polfunnew = update_solve(hat_vr, hat_vd, settings,params,uf)
    econsimnew = ModelSim(params,polfunnew, settings,hf, nsim=100000);
    pdef = round(100 * sum(econsimnew.sim[:, 5])/ 100000; digits = 2);
    display("Simulation finished, with frequency of $pdef default events");

#############################################
# Updating with a NN based on the whole grid
############################################
    hat_vrAct =  0.5*(uvr2-lvr2).*reshape(NNR1act(ss2')',(params.ne,params.nx)) .+ 0.5*(uvr2+lvr2);
    hat_vdAct =  polfun.vd;
    polfunnew = update_solve(hat_vrAct, hat_vdAct, settings,params,uf)
    econsimnew = ModelSim(params,polfunnew, settings,hf, nsim=100000);
    pdef = round(100 * sum(econsimnew.sim[:, 5])/ 100000; digits = 2);
    display("Simulation finished, with frequency of $pdef default events");



    ############################################################
    # Plotting results from the baseline model
    ############################################################
    #=
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
    =#
