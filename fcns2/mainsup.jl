#= ############################################################################
[-] MAIN SUPPORTED CODES: This file has the main supporting codes for
    module of Defaulting Economy. These one are:

    1. ModelSettings: Set the defaults characteristics of a Default Economy
        Its outputs are:
        * Params: A NamedTuple with parameters
        * UtiFun: The utility function, setting as CRRA
        * DefFun: The function h() described in Arellano 2008

    2. FixedPoint: Set some matrices that are no dependent on the Bellman
        operator ann also an educated guess. It has as outputs:
        (in every case it is a matrix of size nₑ x nₓ, where nₑ is the
         amount of points in the bond support and nₓ is the number of
         nodes for the discretization of output)
        * VF: The Value function
        * VC: The continuation value under no default
        * VD: Value of default
        * D:  Default choice
        * BP: Policy functions for issuing of bonds at t+1
        * q:  Price of new issued bonds

    3. value_functions! : It updates the default choices, prices and value
        functions.

    4. updateBellman! : It find the policy function for new issuing debt
        under no-defaulting by maxmizing the expected value function.
        It outputs are:
        * VC : The new continuation value under no default
        * BP : The policy function for new issuing debt

    5. simulation! : Simulates the economy, its output is the matrix Sim with
        the order [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]

############################################################################ =#

function ModelSettings()
    params = (
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
        ub = 0,
        lb = -0.4,
        tol = 1e-8,
        maxite = 1e3,
    )
    utifun = ((x, σrisk) -> (x^(1 - σrisk)) / (1 - σrisk))
    deffun = ((y, fhat) -> min.(y, fhat * mean(y)))
    return params, deffun, utifun
end

function FixedPoint(b, y, udef, P, p0, params, utf)
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat,
        ne, ub, lb, tol, maxite = params;
    dif= 1
    rep= 0
    yb = b .+ y'
    # ----------------------------------------
    # 2. Educated Guess :
    #    vr: value of repayment
    #    vd: value of default
    #    vf: value function
    vr   = 1 / (1 - β) * utf.((r / (1 + r)) * b .+ y', σrisk)
    udef = repeat(udef', ne, 1)
    vd   = 1 / (1 - β) * udef
    vf   = max.(vr, vd)
    D    = 1 * (vd .> vr)
    bb   = repeat(b, 1, nx)
    bp   = Array{CartesianIndex{2},2}(undef, ne, nx)
    q    = Array{Float64,2}(undef, ne, nx)
    # ----------------------------------------
    # 3. Fixed Point Problem
    while dif > tol && rep < maxite
        vf1, vr1, vd1, D1, bp, q=
        value_functions(vf, vr, vd, D, b, P, p0, yb, udef,β, θ, utf, r, σrisk)
        dif = [vf1 vr1 vd1 D1] - [vf vr vd D]
        dif = maximum(abs.(dif))
        vf, vr, vd,D = (vf1,vr1,vd1,D)
        rep += 1
    end
    if rep == maxite
        display("The maximization has not achieved convergence!!!!!!!!")
    else
        print("Convergence achieve after $rep replications \n")
        bp = bb[bp]
    end
    #bp = (1 .- D) .* bp # change 1: if country defaults thge optimal policy is 0
    return vf, vr, vd, D, bp, q
end

function value_functions(vf, vr, vd, D, b, P, p0, yb, udef,β, θ, utf, r, σrisk)
    # ----------------------------------------
    # 1. Saving old information
    ne,nx = size(vf)
    # ----------------------------------------
    # 2. Expected future Value Function
    βevf = β * (vf * P')    # today' value of expected value function
    eδD  = D  * P'          # probability of default in the next period
    evd  = vd * P'          #  expected value of default
    qold = (1 / (1 + r)) * (1 .- eδD) # price
    qb   = qold .* b
    # --------------------------------------------------------------
    # 3. Value function of continuation
    vrnew = Array{Float64,2}(undef,ne,nx)
    bpnew = Array{CartesianIndex{2},2}(undef, ne, nx)
    @inbounds for i = 1:ne
        cc = yb[i, :]' .- qb
        cc[cc.<0] .= 0
        aux_u = utf.(cc, σrisk) + βevf
        vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
    end
    # --------------------------------------------------------------
    # 4. Value function of default
    βθevf = θ * βevf[p0, :]   # expected vf with b=0, in present val
    vdnew    = βθevf' .+ (udef + β * (1 - θ) * evd)
    # --------------------------------------------------------------
    # 5. Continuation Value and Default choice
    vfnew, Dnew = (max.(vrnew, vdnew), 1 * (vdnew .> vrnew))
    return vfnew, vrnew, vdnew, Dnew, bpnew, qold
end

function simulation!(sim, simul_state, PolFun, y, ydef, b,distϕ, nsim2, p0)
    #=
        The next code simulates a economy recursively conditional in a
        predetermined states for the output level (simul_state)
        The algorithm is as follow:
            [1] For each t from 2 to T
                [1.1] Let "Bₜ" the initial level of debt, then we find their
                    position in the grid of b
                [1.2] Let j be the state for output
                [1.3] If the economy is not in default at the start:
                    * Find the default choice Dₜ(Bₜ, jₜ)
                    * Define the output level of the period
                        yⱼ if country does not default
                        yⱼᵈ if does
                    * Define the optimal new issued debt
                        Bₜ₊₁(Bₜ,j) if country does not default
                        0         if does
                [1.4] If the economy is in default at the start
                    * Tose a coin with θ prob for reenter
                    * If they are still out the market:
                        * Bₜ₊₁ = 0 , Dₜ = 1, and VF = VF(0,j), q = q(0,j)
                    * If they re enter to the market
                        * Country makes 1.3 again.
    "[DefStatus,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]"
    =#
    @unpack D, bp, vf , q, vd = PolFun;
    for i = 1:nsim2-1
        bi = findfirst(x -> x == sim[i, 2], b) # position of B
        j = simul_state[i]                     # state for y
        # Choice if there is not previous default
        if sim[i, 1] == 0
            defchoice = D[bi, j]
            ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]
            bsim = (1 - defchoice) * bp[bi, j]
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
                bsim = (1 - defchoice) * bp[p0, j]
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
