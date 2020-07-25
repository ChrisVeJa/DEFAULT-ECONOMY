###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,
    Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux
include("supcodes.jl")
############################################################
# []  Functions
############################################################
############################################################
# SETTING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
          θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
          ub = 0, lb = -0.4, tol = 1e-8, maxite = 500);
uf(x, σrisk)= x^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# Solving
############################################################
function Solver(params, hf, uf)
    # --------------------------------------------------------------
    # 0. Unpacking Parameters
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    # --------------------------------------------------------------
    # 1. Tauchen discretization of log-output
    ly, P = mytauch(μ, ρ, η, nx, m)
    y = exp.(ly)
    # --------------------------------------------------------------
    # 2. Output in case of default
    udef = uf.(hf(y, fhat), σrisk)
    # --------------------------------------------------------------
    # 3. To calculate the intervals of debt I will consider
    grid = (ub - lb) / (ne - 1)
    b = [lb + (i - 1) * grid for i = 1:ne]
    p0 = findmin(abs.(0 .- b))[2]
    b[p0] = 0
    # --------------------------------------------------------------
    # 4. Solving the fixed point problem vf, vr, vd, D, bp, q
    vf, vr, vd, D, bb, q, CC, bp = FixedPoint(b, y, udef, P, p0, params, uf)
    PolFun = (vf = vf, vr = vr, vd = vd, D = D, bb = bb, q = q, CC = CC, bp = bp)
    settings = (P = P, y = y, b= b, udef= udef)
    return PolFun, settings
end

function FixedPoint(b, y, udef, P, p0, params, uf)
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r, σrisk,β, θ, nx, ne, tol, maxite = params;
    dif= 1
    rep= 0
    yb = b .+ y'
    # ----------------------------------------
    vr   = 1 / (1 - β) * uf.((r / (1 + r)) * b .+ y', σrisk)
    udef = repeat(udef', ne, 1)
    vd   = 1 / (1 - β) * udef
    vf   = max.(vr, vd)
    D    = 1 * (vd .> vr)
    bb   = repeat(b, 1, nx)
    bp   = Array{CartesianIndex{2},2}(undef, ne, nx)
    q    = Array{Float64,2}(undef, ne, nx)
    CC   = Array{Float64,2}(undef, ne, nx)
    while dif > tol && rep < maxite
        vf1, vr1, vd1, D1, bp, q, CC = value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
        dif = [vf1 vr1 vd1 D1] - [vf vr vd D]
        dif = maximum(abs.(dif))
        vf, vr, vd, D = (vf1, vr1, vd1, D1)
        rep += 1
    end

    if rep == maxite
        display("The maximization has not achieved convergence!!!!!!!!")
    else
        print("Convergence achieve after $rep replications \n")
        bb = bb[bp]
    end
    return vf, vr, vd, D, bb, q, CC, bp
end

function value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
    @unpack β, θ, r, σrisk, ne, nx = params
    # ----------------------------------------
    # 1. Expected future Value Function
    evf = vf * P'    # today' value of expected value function
    eδD = D  * P'    # probability of default in the next period
    evd = vd * P'    #  expected value of default
    q   = (1 / (1 + r)) * (1 .- eδD) # price
    qb  = q .* b
    # --------------------------------------------------------------
    # 3. Value function of continuation
    vrnew = Array{Float64,2}(undef,ne,nx)
    cnew = Array{Float64,2}(undef,ne,nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    @inbounds for i = 1:ne
        cc = yb[i, :]' .- qb
        cc[cc.<0] .= 0
        aux_u = uf.(cc, σrisk) + β*evf
        vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
        cnew[i, :] = cc[bpnew[i, :]]
    end
    # --------------------------------------------------------------
    # 4. Value function of default
    evaux = θ * evf[p0, :]' .+  (1 - θ) * evd
    vdnew = udef + β*evaux
    # --------------------------------------------------------------
    # 5. Continuation Value and Default choice
    vfnew = max.(vrnew, vdnew)
    Dnew = 1 * (vdnew .> vrnew)
    eδD  = Dnew  * P'
    qnew = (1 / (1 + r)) * (1 .- eδD)
    return vfnew, vrnew, vdnew, Dnew, bpnew, qnew, cnew
end


polfun, settings = Solver(params, hf, uf);

############################################################
# Simulation
############################################################
function ModelSim(params, PolFun, settings, hf; nsim = 100000, burn = 0.05)
    # -------------------------------------------------------------------------
    # 0. Settings
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    P = settings.P
    b = settings.b
    y = settings.y
    ydef = hf(y, fhat)
    p0 = findmin(abs.(0 .- b))[2]
    nsim2 = Int(floor(nsim * (1 + burn)))
    # -------------------------------------------------------------------------
    # 1. State simulation
    choices = 1:nx    # Possible states
    # if nseed != 0 Random.seed!(nseed[1]) end
    simul_state = zeros(Int64, nsim2);
    simul_state[1]  = rand(1:nx);
    for i = 2:nsim2
        simul_state[i] =
            sample(view(choices, :, :), Weights(view(P, simul_state[i-1], :)))
    end

    # -------------------------------------------------------------------------
    # 2. Simulation of the Economy
    orderName = "[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) yⱼ"
    distϕ = Bernoulli(θ)
    EconSim = Array{Float64,2}(undef, nsim2, 8)  # [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
    EconSim[1, 1:2] = [0 b[rand(1:ne)]]  # b could be any value in the grid
    EconSim = simulation!(
        EconSim,simul_state,PolFun,y,ydef,b,distϕ,nsim2,p0)
    # -------------------------------------------------------------------------
    # 3. Burning and storaging
    EconSim = EconSim[end-nsim:end-1, :]
    modelsim = (sim = EconSim, order = orderName)
    return modelsim
end

function simulation!(sim, simul_state, PolFun, y, ydef, b,distϕ, nsim2, p0)
    @unpack D, bb, vf , q, vd = PolFun;
    for i = 1:nsim2-1
        bi = findfirst(x -> x == sim[i, 2], b) # position of B
        j = simul_state[i]                     # state for y
        # Choice if there is not previous default
        if sim[i, 1] == 0
            defchoice = D[bi, j]
            ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]
            bsim = (1 - defchoice) * bb[bi, j]
            sim[i, 3:8] =
                [ysim bsim defchoice vf[bi, j] q[bi, j] y[j]]
            sim[i+1, 1:2] = [defchoice bsim]
        else
            # Under previous default, I simulate if the economy could reenter to the market
            defstat = rand(distϕ)
            if defstat == 1 # They are in the market
                sim[i, 1] == 0
                defchoice = D[p0, j] # default again?
                ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]# output | choice
                bsim = (1 - defchoice) * bb[p0, j]
                sim[i, 3:8] =
                    [ysim bsim defchoice vf[p0, j] q[p0,j] y[j]]
                sim[i+1, 1:2] = [defchoice bsim]
            else # They are out the market
                sim[i, 3:8] =
                    [ydef[j] 0 1 vd[p0, j] q[p0, j] y[j]] #second change
                sim[i+1, 1:2] = [1 0]
            end
        end
    end
    return sim
end

econsim = ModelSim(params,polfun, settings,hf);
pdef = round(100 * sum(econsim.sim[:, 5])/ 100000; digits = 2);
display("Simulation finished, with frequency of $pdef default events");


############################################################
# Training
############################################################
mynorm(x) = begin
    ux = maximum(x, dims=1)
    lx = minimum(x, dims=1)
    nx = (x .- 0.5(ux+lx)) ./ (0.5*(ux-lx))
    return nx, ux, lx
end
y  = econsim.sim[:,8];
ys, uys, lys = mynorm(y);
bs = econsim.sim[:,2];
ss = [bs ys]
bb = econsim.sim[:,4];
Q1    = 16
NNB   = Chain(Dense(2, Q1, softplus), Dense(Q1, 1))
lossb(x, y) = Flux.mse(NNB(x), y)
datab = Flux.Data.DataLoader(ss', bb')
psb   = Flux.params(NNB)
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossb, psb, datab, Descent())
   display(lossb(ss', bb'))
end
dplot = [bb NNB(ss')'];
plot(dplot[1:500,:], label = ["bond" "NN"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red], alpha = 0.7, w = [1.15 0.75], legend=:bottomright, grid=:false)

def = econsim.sim[:,5];
NND =
lossD = 
datad = Flux.Data.DataLoader(ss', def')
psd  = Flux.params(NND)
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossd, psd, datad, Descent())
   display(lossd(ss', def'))
end
dplot = [def NND(ss')'];
plot(dplot[1:1000,:], label = ["bond" "NN"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red], alpha = 0.7, w = [1.15 0.75], legend=:bottomright, grid=:false)



plot(def[1:2000])
#=
 Graphics for fit
=#
heatmap(settings.y, settings.b, polfun.D', aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" )





D = polfun.D
bb = polfun.bb
bp = polfun.bp

fixedp(bb,D,bp, settings, params, uf) = begin
    @unpack P, y, b,udef   = settings
    @unpack σrisk, β, ne,r, θ = params
    vd = 1/(1-β) * udef;
    vd = repeat(vd',ne, 1)
    vr = 1/(1 - β) * uf.((r / (1 + r)) * b .+ y', σrisk)
    q  = (1 / (1 + r)) * (1 .- (D*P'))
    v   = max.(vr,vd)
    cc  = (b .+ y') - (q[bp].*bb);
    cc[cc.<0] .= 0;
    uvr = uf.(cc,σrisk)
    dif = 1.0
    iteration = 1
    while dif > 1e-8 && iteration<10000
        v1, vr1, vd1 , dif = updated!(v,vr, vd,D, uvr, udef,params,bp, P)
        vr = vr1; vd=  vd1 ; v = v1;
        iteration+=1
    end
    return v,vr,vd, iteration
end

updated!(v, vr,vd, D, uvr, udef, params, bp, P) = begin
    @unpack β, θ = params
    ev  = v * P'
    evd = vd * P'
    vr1 = uvr + β*ev[bp]
    evaux =  θ * ev[end, :]' .+  (1 - θ) * evd
    vd1 = udef' .+ β*evaux
    v1  = max.(vr1,vd1)
    dif = max(maximum(abs.(v1-v)),maximum(abs.(vr1-vr)), maximum(abs.(vd1-vd)))
    return v1, vr1, vd1, dif
end

polfun.
vg,vrg,vdg, iteration = fixedp(bb,D,bp, settings, params, uf)
display(maximum(abs.(vdg-polfun.vd)))
