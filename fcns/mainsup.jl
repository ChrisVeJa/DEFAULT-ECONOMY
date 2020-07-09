###############################################################################
# [] MAIN SUPPORTED CODES
###############################################################################

# =============================================================================
# [1] Default settings
# =============================================================================
function ModelSettings()
    Params = (
        r = 0.017,
        σrisk = 2.0,
        ρ = 0.945,
        η = 0.025,
        β = 0.953,
        θ = 0.282,
        nx = 21,
        m = 3,
        μ = 0.0,
        fhat = 0.969,
        ne = 251,
        ub = 0.4,
        lb = -0.4,
        tol = 1e-8,
        maxite = 1e3,
    )
    UtiFun = ((x, σrisk) -> (x^(1 - σrisk)) / (1 - σrisk))
    DefFun = ((y, fhat) -> min.(y, fhat * mean(y)))
    return Params, DefFun, UtiFun
end

# =============================================================================
# [2] Fixed Point Problem
# =============================================================================
function FixedPoint(b, y, udef, pix, posb0, Params, utf)
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol, maxite =
        Params
    dif = 1
    rep = 0
    yb = b .+ y'
    # ----------------------------------------
    # 2. Educated Guess
    VC = 1 / (1 - β) * utf.((r / (1 + r)) * b .+ y', σrisk)
    udef = repeat(udef', ne, 1)
    VD = 1 / (1 - β) * udef
    VO = max.(VC, VD)
    D = 1 * (VD .> VC)
    BB = repeat(b, 1, nx)
    Bprime = Array{CartesianIndex{2},2}(undef, ne, nx)
    q = Array{Float64,2}(undef, ne, nx)
    # ----------------------------------------
    # 3. Fixed Point Problem
    while dif > tol && rep < maxite
        VO, VC, VD, D, Bprime, q, dif = value_functions!(
            VO,
            VC,
            VD,
            D,
            Bprime,
            q,
            dif,
            b,
            pix,
            posb0,
            yb,
            udef,
            β,
            θ,
            utf,
            r,
            σrisk,
        )
        rep += 1
    end
    if rep == maxite
        display("The maximization has not achieved convergence!!!!!!!!")
    else
        print("Convergence achieve after $rep replications \n")
        Bprime = BB[Bprime]
    end
    return VO, VC, VD, D, Bprime, q
end

# =============================================================================
# [2] Updating the value functions
# =============================================================================
function value_functions!(
    VO,
    VC,
    VD,
    D,
    Bprime,
    q,
    dif,
    b,
    pix,
    posb0,
    yb,
    udef,
    β,
    θ,
    utf,
    r,
    σrisk,
)
    # ----------------------------------------
    # 1. Saving old information
    VOold = VO
    ne, nx = size(VO)
    # ----------------------------------------
    # 2. Expected future Value Function
    EVC = VC * pix'
    EδD = D * pix'
    EVD = VD * pix'
    βEV = β * EVC
    qold = (1 / (1 + r)) * (1 .- EδD)
    qB = qold .* b
    # --------------------------------------------------------------
    # 3. Value function of continuation
    VC, Bprime = updateBellman!(VC, Bprime, yb, qB, βEV, utf, σrisk, ne, nx)
    # --------------------------------------------------------------
    # 4. Value function of default
    βθEVC0 = β * θ * EVC[posb0, :]
    VD = βθEVC0' .+ (udef + β * (1 - θ) * EVD)
    # --------------------------------------------------------------
    # 5. Continuation Value and Default choice
    VO, D = (max.(VC, VD), 1 * (VD .> VC))
    q = (1 / (1 + r)) * (1 .- (D * pix'))
    # --------------------------------------------------------------
    # 6.  Divergence respect the initial point
    dif = maximum(abs.(VO - VOold))
    return VO, VC, VD, D, Bprime, q, dif
end

# =============================================================================
# [3] Solving the Bellman operator under No Default
# =============================================================================
function updateBellman!(
    VC,
    Bprime,
    yb,
    qB,
    βEV,
    utf::Function,
    σrisk::Float64,
    ne::Int64,
    nx::Int64,
)
    @inbounds for i = 1:ne
        cc = yb[i, :]' .- qB
        cc[cc.<0] .= 0
        aux_u = utf.(cc, σrisk) + βEV
        VC[i, :], Bprime[i, :] = findmax(aux_u, dims = 1)
    end
    return VC, Bprime
end

# =============================================================================
# [4] Loop for simulation
# =============================================================================
function simulation!(
    EconSim,
    simul_state,
    EconBase,
    y,
    ydef,
    b,
    distϕ,
    nsim2,
    posb0,
)
    for i = 1:nsim2-1
        bi = findfirst(x -> x == EconSim[i, 2], b)
        j = simul_state[i]
        # Choice if there is not previous default
        if EconSim[i, 1] == 0
            defchoice = EconBase.D[bi, j]
            ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]
            bsim = (1 - defchoice) * EconBase.BP[bi, j]
            EconSim[i, 3:7] =
                [ysim bsim defchoice EconBase.VF[bi, j] EconBase.q[bi, j]]
            EconSim[i+1, 1:2] = [defchoice bsim]
        else
            # Under previous default, I simulate if the economy could reenter to the market
            defstat = rand(distϕ)
            if defstat == 1 # They are in the market
                defchoice = EconBase.D[posb0, j] # default again?
                ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]# output | choice
                bsim = (1 - defchoice) * EconBase.BP[posb0, j]
                EconSim[i, 3:7] =
                    [ysim bsim defchoice EconBase.VF[posb0, j] EconBase.q[
                        posb0,
                        j,
                    ]]
                EconSim[i+1, 1:2] = [defchoice bsim]
            else # They are out the market
                EconSim[i, 3:7] =
                    [ydef[j] 0 1 EconBase.VF[posb0, j] EconBase.q[posb0, j]]
                EconSim[i+1, 1:2] = [1 0]
            end
        end
    end
    return EconSim
end
# =============================================================================
