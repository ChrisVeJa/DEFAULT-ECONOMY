###############################################################################
# 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
###############################################################################

############################################################
# [0] Including our module
############################################################

using Random,Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux;
include("supcodes.jl")
############################################################
# []  Functions
############################################################
############################################################
#           [] SETTING - SOLVING - SIMULATING
############################################################
params = (r = 0.017, σrisk = 2.0, ρ = 0.945, η = 0.025, β = 0.953,
          θ = 0.282, nx = 21, m = 3, μ = 0.0,fhat = 0.969, ne = 251,
          ub = 0, lb = -0.4, tol = 1e-8, maxite = 500);
uf(x, σrisk)= x^(1 - σrisk) / (1 - σrisk)
hf(y, fhat) = min.(y, fhat * mean(y))


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
    # 4. Solving the fixed point problem
    vf, vr, vd, D, bp, q = FixedPoint(b, y, udef, P, p0, params, uf)
    PolFun = (vf = vf, vr = vr, vd = vd, D = D, bp = bp, q = q)
    return PolFun
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

    while dif > tol && rep < maxite
        vf1, vr1, vd1, D1, bp, q = value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
        dif = vf1 - vf
        dif = maximum(abs.(dif))
        vf, vr, vd, D = (vf1, vr1, vd1, D)
        rep += 1
    end

    if rep == maxite
        display("The maximization has not achieved convergence!!!!!!!!")
    else
        print("Convergence achieve after $rep replications \n")
        bp = bb[bp]
    end
    return vf1, vr1, vd1, D1, bp, q
end

function value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
    @unpack β, θ, r, σrisk, ne, nx = params
    # ----------------------------------------
    # 1. Expected future Value Function
    βevf = β * (vf * P')    # today' value of expected value function
    eδD  = D  * P'          # probability of default in the next period
    evd  = vd * P'          #  expected value of default
    q    = (1 / (1 + r)) * (1 .- eδD) # price
    qb   = q .* b
    # --------------------------------------------------------------
    # 3. Value function of continuation
    vrnew = Array{Float64,2}(undef,ne,nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    @inbounds for i = 1:ne
        cc = yb[i, :]' .- qb
        cc[cc.<0] .= 0
        aux_u = uf.(cc, σrisk) + βevf
        vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
    end
    # --------------------------------------------------------------
    # 4. Value function of default
    evaux = θ * βevf[p0, :]' .+  (1 - θ) * evd
    vdnew = udef + β*(evaux)
    # --------------------------------------------------------------
    # 5. Continuation Value and Default choice
    vfnew = max.(vrnew, vdnew)
    Dnew  = 1 * (vdnew .> vrnew))
    return vfnew, vrnew, vdnew, Dnew, bpnew, q
end

#=
 Solving
=#
econdef = Solver(params, hdef, uf);
polfun = econdef.PolFun;
ext = econdef.ext;

ev  = polfun.vf * ext.P';
evd = polfun.vd * ext.P';
cc  = (ext.bgrid .+ ext.ygrid') - (polfun.q .* polfun.bp)
vr1 = uf.(cc, params.σrisk) + params.β * ev
βθevf = params.θ * params.β * ev[end, :]
vd1 = (βθevf +  uf.(ext.ydef, params.σrisk))'  .+ params.β * (1 - params.θ) * evd
vf1 = max.(vr1,vd1)
display(maximum(abs.(vf1 - polfun.vf)))
# --------------------------------------------------------------
# 5. Continuation Value and Default choice
vf, D = (max.(vr, vd), 1 * (vd .> vr))
q = (1 / (1 + r)) * (1 .- (D * P'))
updated!(VR, VD, V, D, uvr, uvd, params, ext) = begin
   eV  = V * ext.P'
   VR1 = uvr + params.β*eV
   ev0 = eV[end,:]
   VD1 = (uvd + params.β*params.θ * ev0)' .+ params.β*(1-params.θ)*(VD*ext.P')
   V1 , D1  = (max.(VD1,VR1), 1 .* (VD1 .> VR1))
   dif = max(maximum(abs.(VR1-VR)),maximum(abs.(V1-V)), maximum(abs.(VD1-VD)),maximum(abs.(D1-D)))
   return VR1, VD1, V1, dif
end

fixedp(params, polfun, ext, uf) = begin
   uvd = uf.(ext.ydef,params.σrisk)
   VD  = 1/(1-params.β) * uvd;
   VD  = repeat(VD',params.ne, 1)
   VR  = 1/(1 - params.β) * uf.((params.r / (1 + params.r)) * ext.bgrid .+ ext.ygrid', params.σrisk)
   V   = max.(VD,VR)
   D   = polfun.D
   qbp = polfun.q .* polfun.bp;
   cc  = (ext.bgrid .+ ext.ygrid') - qbp;
   cc[cc.<0] .= 0;
   uvr = uf.(cc,params.σrisk)
   dif = 1.0
   iteration = 1
   while dif > 1e-8 && iteration<10000
      VR1, VD1, V1, dif = updated!(VR, VD, V,D,uvr, uvd,params,ext)
      VR = VR1; VD =  VD1 ; V = V1;
      iteration+=1
   end
   return VR, VD, V,dif, iteration
end
VR, VD, V, dif, iteration = fixedp(params, polfun, ext, uf)
