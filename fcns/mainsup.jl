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

function FixedPoint(b, y, udef, P, p0, Params, utf)
    #=
        The inputs for this function are:
            * b: the grid for the bond support
            * y: the discretization of y
            * udef: The utility function for default.
                It is a vector not a matrix
            * P: the markov chain matrix
            * p0: The position for b=0 in the support
            * Params: A NamedTuple with the parameters
            * utf: The utility function
        The function is structured as follow
            [1] Calculate fixed matrices
                yb = b .+ y >> Matrix nₑ x nₓ
                The utility function  u(y + B -qB')
                has y .+ B (with B being all the possibles)
                which does not depend on B'
            [2] The educated guesses:
                VC₀ = (1-β)⁻¹ u(r/(1+r) * b + y)
                    never defaulting and receiving the annualities for b
                VD₀ = (1-β)⁻¹ u(yᵈ)
                    being in default forever
            [3] Fixed Point Problem
    =#
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat,
        ne, ub, lb, tol, maxite = Params;
    dif= 1
    rep= 0
    yb = b .+ y'
    # ----------------------------------------
    # 2. Educated Guess
    VC = 1 / (1 - β) * utf.((r / (1 + r)) * b .+ y', σrisk)
    udef = repeat(udef', ne, 1)
    VD = 1 / (1 - β) * udef
    VF = max.(VC, VD)
    D  = 1 * (VD .> VC)
    BB = repeat(b, 1, nx)
    BP = Array{CartesianIndex{2},2}(undef, ne, nx)
    q  = Array{Float64,2}(undef, ne, nx)
    # ----------------------------------------
    # 3. Fixed Point Problem
    while dif > tol && rep < maxite
        VF, VC, VD, D, BP, q, dif = value_functions!(
            VF, VC, VD, D, BP, q, dif, b, P, p0,
            yb, udef, β, θ, utf, r, σrisk
        )
        rep += 1
    end
    if rep == maxite
        display("The maximization has not achieved convergence!!!!!!!!")
    else
        print("Convergence achieve after $rep replications \n")
        BP = BB[BP]
    end
    BP = (1 .-D) .* BP # change 1: if country defaults thge optimal policy is 0
    return VF, VC, VD, D, BP, q
end

function value_functions!( VF, VC, VD, D, BP, q, dif, b, P, p0, yb, udef,
            β, θ, utf, r, σrisk)
    #=
        The structure of the function is:
            [1] Calculate E[x] where x is the value function
            [2] Find the optimal issued bond level under no default
            [3] Update the value of default
            [4] Update the new value function and default decision
                as well the consistent prices
    =#
    # ----------------------------------------
    # 1. Saving old information
    VFold = VF
    ne,nx = size(VF)
    # ----------------------------------------
    # 2. Expected future Value Function
    EVC = VC * P'
    EδD = D * P'
    EVD = VD * P'
    βEV = β * EVC
    qold = (1 / (1 + r)) * (1 .- EδD)
    qB = qold .* b
    # --------------------------------------------------------------
    # 3. Value function of continuation
    VC, BP = updateBellman!(VC, BP, yb, qB, βEV, utf, σrisk, ne, nx)
    # --------------------------------------------------------------
    # 4. Value function of default
    βθEVC0 = β * θ * EVC[p0, :]
    VD = βθEVC0' .+ (udef + β * (1 - θ) * EVD)
    # --------------------------------------------------------------
    # 5. Continuation Value and Default choice
    VF, D = (max.(VC, VD), 1 * (VD .> VC))
    q = (1 / (1 + r)) * (1 .- (D * P'))
    # --------------------------------------------------------------
    # 6.  Divergence respect the initial point
    dif = maximum(abs.(VF - VFold))
    return VF, VC, VD, D, BP, q, dif
end

function updateBellman!(VC, BP, yb, qB, βEV, utf, σrisk, ne, nx )
    #=
        This function find the value of Bₜ₊₁ that maximizes the expected
        value function. The algorithm is:
            [1] For each level of current debt (Bₜ) calculates
                [1.1] calculates the consumption: c= y + B - qB'
                [1.2] if c<0 we put c==0
                [1.3] for each possible new debt level calculate u(c) + βE[V]
                [1.4] Find the level of new debt with maximum current value

    =#
    @inbounds for i = 1:ne
        cc = yb[i, :]' .- qB
        cc[cc.<0] .= 0
        aux_u = utf.(cc, σrisk) + βEV
        VC[i, :], BP[i, :] = findmax(aux_u, dims = 1)
    end
    return VC, BP
end

function simulation!(Sim, simul_state, EconBase, y, ydef, b,distϕ, nsim2, p0)
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
    for i = 1:nsim2-1
        bi = findfirst(x -> x == Sim[i, 2], b)
        j = simul_state[i]
        # Choice if there is not previous default
        if Sim[i, 1] == 0
            defchoice = EconBase.D[bi, j]
            ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]
            bsim = (1 - defchoice) * EconBase.BP[bi, j]
            Sim[i, 3:8] =
                [ysim bsim defchoice EconBase.VF[bi, j] EconBase.q[bi, j] j]
            Sim[i+1, 1:2] = [defchoice bsim]
        else
            # Under previous default, I simulate if the economy could reenter to the market
            defstat = rand(distϕ)
            if defstat == 1 # They are in the market
                Sim[i, 1] == 0
                defchoice = EconBase.D[p0, j] # default again?
                ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]# output | choice
                bsim = (1 - defchoice) * EconBase.BP[p0, j]
                Sim[i, 3:8] =
                    [ysim bsim defchoice EconBase.VF[p0, j] EconBase.q[p0,j] j]
                Sim[i+1, 1:2] = [defchoice bsim]
            else # They are out the market
                Sim[i, 3:8] =
                    [ydef[j] 0 1 EconBase.VD[p0, j] EconBase.q[p0, j] j] #second change
                Sim[i+1, 1:2] = [1 0]
            end
        end
    end
    return Sim
end
