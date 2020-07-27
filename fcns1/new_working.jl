###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random, Optim, Distributions, Statistics, LinearAlgebra, Plots,
    StatsBase, Parameters, Flux
include("supcodes.jl")
############################################################
# []  Functions
############################################################
############################################################
# SETTING
############################################################
#params1 = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
#          θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
#          ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 251);
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 501);
#params3 = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
#          θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
#          ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 1001);
#params4 = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
#        θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969,
#        ub = 0, lb = -0.4, tol = 1e-8, maxite = 500, ne = 2001);
uf(x)= x^(1 - params.σrisk) / (1 - params.σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))

############################################################
# Solving
############################################################
function Solver(params, hf, uf)
    # --------------------------------------------------------------
    # 0. Unpacking Parameters
    @unpack r, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    # --------------------------------------------------------------
    # 1. Tauchen discretization of log-output
    ly, P = mytauch(μ, ρ, η, nx, m)
    y = exp.(ly)
    # --------------------------------------------------------------
    # 2. Output in case of default
    udef = uf.(hf(y, fhat))
    # --------------------------------------------------------------
    # 3. To calculate the intervals of debt I will consider
    grid = (ub - lb) / (ne - 1)
    b = [lb + (i - 1) * grid for i = 1:ne]
    p0 = findmin(abs.(0 .- b))[2]
    b[p0] = 0
    # --------------------------------------------------------------
    # 4. Solving the fixed point problem vf, vr, vd, D, bp, q
    vf, vr, vd, D, bb, q, bp = FixedPoint(b, y, udef, P, p0, params, uf)
    PolFun = (vf = vf, vr = vr, vd = vd, D = D, bb = bb, q = q,bp = bp)
    settings = (P = P, y = y, b= b, udef= udef)
    return PolFun, settings
end

function FixedPoint(b, y, udef, P, p0, params, uf)
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r,β, θ, nx, ne, tol, maxite = params;
    dif= 1
    rep= 0
    yb = b .+ y'
    # ----------------------------------------
    vr   = 1 / (1 - β) * uf.((r / (1 + r)) * b .+ y')
    udef = repeat(udef', ne, 1)
    vd   = 1 / (1 - β) * udef
    vf   = max.(vr, vd)
    D    = 1 * (vd .> vr)
    bb   = repeat(b, 1, nx)
    bp   = Array{CartesianIndex{2},2}(undef, ne, nx)
    q    = Array{Float64,2}(undef, ne, nx)
    CC   = Array{Float64,2}(undef, ne, nx)
    while dif > tol && rep < maxite
        vf1, vr1, vd1, D1, bp, q = value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
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
    return vf, vr, vd, D, bb, q, bp
end

function value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
    @unpack β, θ, r, ne, nx = params
    # ----------------------------------------
    # 1. Expected future Value Function
    evf = vf * P'    # today' value of expected value function
    eδD = D  * P'    # probability of default in the next period
    evd = vd * P'    #  expected value of default
    q   = (1 / (1 + r)) * (1 .- eδD) # price
    qb  = q .* b
    βevf= β*evf
    # --------------------------------------------------------------
    # 3. Value function of continuation
    vrnew = Array{Float64,2}(undef,ne,nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    mybellman!(vrnew,bpnew,yb, qb,βevf, uf,ne)
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
    return vfnew, vrnew, vdnew, Dnew, bpnew, qnew
end
function mybellman!(vrnew,bpnew,yb, qb,βevf, uf,ne)
    @inbounds for i = 1:ne
    cc = yb[i, :]' .- qb
    cc = max.(cc,0)
    #aux_u = uf.(cc, σrisk) + βevf
    aux_u = uf.(cc)  + βevf
    vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
    end
    return vrnew, bpnew
end

#@time polfun1, settings1 = Solver(params1, hf, uf);
@time polfun2, settings2 = Solver(params2, hf, uf);
#@time polfun3, settings3 = Solver(params3, hf, uf);
#@time polfun4, settings4 = Solver(params4, hf, uf);
#heat1 = heatmap(settings1.y, settings1.b, polfun1.D', aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
heat2 = heatmap(settings2.y, settings2.b, polfun2.D',
        aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
#heat3 = heatmap(settings3.y, settings3.b, polfun3.D', aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
#heat4 = heatmap(settings4.y, settings4.b, polfun4.D', aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
#plot(heat1,heat2,heat3,heat4, layout=(2,2))
#savefig("./Figures/heatmap.svg")
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

DD  = -ones(params.ne,params.nx)
sts =  econsim.sim[:,2:3]
stuple = [(sts[i,1], sts[i,2])  for i in 1:size(sts)[1] ];
for i in 1:params.ne
    bi = settings.b[i]
    for j in 1:params.nx
        yj = settings.y[j]
        pos = findfirst(x -> x==(bi,yj), stuple)
        if ~isnothing(pos)
            DD[i,j] = econsim.sim[pos, 5]
        end
    end
end
heat2 = heatmap(settings.y, settings.b, DD', c = cgrad([:white, :black, :yellow]),aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
compheat = plot(heat1,heat2,layout=(2,1),size =(800,1000));
savefig("./Figures/compheat.svg")
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
ss = [-bs ys]
bb = -econsim.sim[:,4];
Q1    = 16
NNB   = Chain(Dense(2, Q1, softplus), Dense(Q1, 1, sigmoid))
lossb(x, y) = Flux.mse(NNB(x), y)
datab = Flux.Data.DataLoader(ss', bb')
psb   = Flux.params(NNB)
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossb, psb, datab, Descent())
   display(lossb(ss', bb'))
end
dplot = [bb NNB(ss')'];
plot(dplot[1:500,:], label = ["bond" "NN"], fg_legend=:transparent,bg_legend=:transparent,
    c=[:blue :red], alpha = 0.7, w = [1.15 0.75], legend=:topright, grid=:false)

#=
dd = econsim.sim[:,5];
Q1 = 8
NND = Chain(Dense(2,Q1,softplus), Dense(Q1,1,sigmoid));
lossd(x::Array,y::Array) = begin
     x1 = NND(x)
     logll = log.(x1 .+ 0.0000001).*y + log.(1.0000001 .- x1) .* (1 .-y)
     logll = mean(logll)
     return -logll
end
ss1   = Array{Float32}(ss')
dd1   = Array{Float32}(dd')
datad = Flux.Data.DataLoader(ss1,dd1)
psd   = Flux.params(NND);
Flux.@epochs 10 begin
   Flux.Optimise.train!(lossd, psd, datad, ADAM())
   display(lossd(ss1, dd1))
end
lik = NND(ss')';
cgrid = maximum(lik)/100
td = size(lik)[1]
=#



pp(x,b) = begin
    xb = x * b'
    return (1 .+ exp.(-xb)) .^ (-1)
end

function lllog(b,x,y)
    xb = x * b'
    fxb = (1 .+ exp.(-xb)) .^ (-1)
    LL  = y .* log.(fxb)  + (1 .- y) .* log.(1 .- fxb)
    return -sum(LL)
end
def = econsim.sim[:,5];
mylln(beta) = lllog(beta,ss,def);
betas  = [0.0 0.0];
result = optimize(mylln, betas, BFGS())
predict = pp(ss, result.minimizer)

cut0 = 0
global disO = 1
objt = mean(def)
for i in 1:100
    cutN = i/100
    yhat = 1 * (predict .> cutN)
    disN =  abs.(mean(yhat) - objt)
    if disN < disO
        global disO = disN
        global cut0 = cutN
    end
end
yhat = 1 * (predict .> cut0);

b1  = -settings.b
y1 , uys, lys = mynorm(settings.y);
states  = [repeat(b1,params.nx,1) repeat(y1, inner= params.ne)]
Dsolhat = pp(states, result.minimizer)
Dsolhat = reshape(Dsolhat, (params.ne,params.nx))
Dhat    = 1 * (Dsolhat .> cut0);
heat3   = heatmap(settings.y, settings.b, Dhat', aspect_ratio = 0.8, xlabel = "Output", ylabel = "Debt" );
plot(heat1,heat2, heat3,layout=(3,1),size =(1200,1000), aspect_ratio = 0.6)



#=
 Graphics for fit
=#





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
