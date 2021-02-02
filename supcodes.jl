function Solver(params, hf, uf)
    # --------------------------------------------------------------
    # 0. Unpacking Parameters
    @unpack r, ρ, η, β, θ, σrisk, nx, m, μ, fhat, ne, ub, lb, tol = params
    # --------------------------------------------------------------
    # 1. Tauchen discretization of log-output
    ly, P = mytauch(μ, ρ, η, nx, m)
    y = exp.(ly)
    # --------------------------------------------------------------
    # 2. Output in case of default
    udef = uf(hf(y, fhat), σrisk)
    # --------------------------------------------------------------
    # 3. To calculate the intervals of debt I will consider
    grid = (ub - lb) / (ne - 1)
    b = [lb + (i - 1) * grid for i = 1:ne]
    p0 = findmin(abs.(0 .- b))[2]
    b[p0] = 0
    # --------------------------------------------------------------
    # 4. Solving the fixed point problem vf, vr, vd, D, bp, q
    vf, vr, vd, D, bb, q, bp, eδD = FixedPoint(b, y, udef, P, p0, params, uf)
    PolFun = (vf = vf, vr = vr, vd = vd, D = D, bb = bb, q = q,bp = bp,eδD =eδD)
    settings = (P = P, y = y, b= b, udef= udef)
    return PolFun, settings
end
function FixedPoint(b, y, udef, P, p0, params, uf)
    # ----------------------------------------
    # 1. Some initial parameters
    @unpack r,β, θ, σrisk, nx, ne, tol, maxite = params;
    dif= 1
    rep= 0
    yb = b .+ y'
    # ----------------------------------------
    vr   = 1 / (1 - β) * uf((r / (1 + r)) * b .+ y', σrisk)
    udef = repeat(udef', ne, 1)
    vd   = 1 / (1 - β) * udef
    vf   = max.(vr, vd)
    D    = 1 * (vd .> vr)
    bb   = repeat(b, 1, nx)
    bp   = Array{CartesianIndex{2},2}(undef, ne, nx)
    q    = Array{Float64,2}(undef, ne, nx)
    CC   = Array{Float64,2}(undef, ne, nx)
    eδD = nothing
    while dif > tol && rep < maxite
        vf1, vr1, vd1, D1, bp, q, eδD = value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
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
    return vf, vr, vd, D, bb, q, bp, eδD
end
function value_functions(vf, vr, vd, D, b, P, p0, yb, udef,params,uf)
    @unpack β, θ, r, ne, nx, σrisk = params
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
    mybellman!(vrnew,bpnew,yb, qb,βevf, uf,ne, σrisk)
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
    return vfnew, vrnew, vdnew, Dnew, bpnew, qnew, eδD
end
function mybellman!(vrnew,bpnew,yb, qb,βevf, uf,ne, σrisk)
    @inbounds for i = 1:ne
    cc = yb[i, :]' .- qb
    cc = max.(cc,0)
    aux_u = uf.(cc, σrisk) + βevf
    vrnew[i, :], bpnew[i, :] = findmax(aux_u, dims = 1)
    end
    return vrnew, bpnew
end
function ModelSim(params, PolFun, settings, hf; nsim = 100000, burn = 0.05, ini_st = 0)
    # -------------------------------------------------------------------------
    # 0. Settings
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    P = settings.P
    b = settings.b
    y = settings.y
    ydef = hf(y, fhat)
    p0 = findmin(abs.(0 .- b))[2]
    nsim2 = Int(floor(nsim * (1 + burn)))+1
    # -------------------------------------------------------------------------
    # 1. State simulation
    choices = 1:nx    # Possible states
    # if nseed != 0 Random.seed!(nseed[1]) end
    simul_state = zeros(Int64, nsim2);
    if ini_st == 0
        simul_state[1]  = rand(1:nx);
    else
        simul_state[1]  = ini_st[1];
    end
    for i = 2:nsim2
        simul_state[i] =
            sample(view(choices, :, :), Weights(view(P, simul_state[i-1], :)))
    end
    #=mc = MarkovChain(P)
    if ini_st == 0
        simul_state = simulate(mc, nsim2, init = rand(1:nx));
    else
        simul_state = simulate(mc, nsim2, init = ini_st[1]);
    end=#


    # -------------------------------------------------------------------------
    # 2. Simulation of the Economy
    orderName = "[Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ)) yⱼ vr vd θ"
    distϕ = Bernoulli(θ)
    EconSim = Array{Float64,2}(undef, nsim2, 11)  # [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]
    if ini_st == 0
        EconSim[1, 1:2] = [0 b[rand(1:ne)]]  # b could be any value in the grid
    else
        EconSim[1, 1:2] = [ini_st[2] b[ini_st[3]]]  # b could be any value in the grid
    end

    EconSim = simulation!(
        EconSim,simul_state,PolFun,y,ydef,b,distϕ,nsim2,p0)
    # -------------------------------------------------------------------------
    # 3. Burning and storaging
    EconSim = EconSim[end-nsim:end-1, :]
    modelsim = (sim = EconSim, order = orderName)
    return modelsim
end
function simulation!(sim, simul_state, PolFun, y, ydef, b,distϕ, nsim2, p0)
    @unpack D, bb, vf , q, vd,vr = PolFun;
    for i = 1:nsim2-1
        bi = findfirst(x -> x == sim[i, 2], b) # position of B
        j = simul_state[i]                     # state for y
        # Choice if there is not previous default
        if sim[i, 1] == 0
            defchoice = D[bi, j]
            ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]
            bsim = (1 - defchoice) * bb[bi, j]
            sim[i, 3:11] =
                [ysim bsim defchoice vf[bi, j] q[bi, j] y[j]  vr[bi, j]  vd[bi, j] 0]
            sim[i+1, 1:2] = [defchoice bsim]
        else
            # Under previous default, I simulate if the economy could reenter to the market
            defstat = rand(distϕ)
            if defstat == 1 # They are in the market
                sim[i, 1] == 0
                defchoice = D[p0, j] # default again?
                ysim = (1 - defchoice) * y[j] + defchoice * ydef[j]# output | choice
                bsim = (1 - defchoice) * bb[p0, j]
                sim[i, 3:11] =
                    [ysim bsim defchoice vf[p0, j] q[p0,j] y[j] vr[p0, j] vd[p0, j] 0]
                sim[i+1, 1:2] = [defchoice bsim]
            else # They are out the market
                sim[i, 3:11] =
                    [ydef[j] 0 1 vd[p0, j] q[p0, j] y[j] vr[p0, j] vd[p0, j] 1] #second change
                sim[i+1, 1:2] = [1 0]
            end
        end
    end
    return sim
end
###############################################################################
# [] CODE FOR GRAPHICS
###############################################################################
function graph_solve(
    PolFun,
    ext;
    titles = ["BondPrice.svg" "Savings.svg" "ValFun.svg"],
    )
    # --------------------------------------------------------------
    # 0. Unpacking
    @unpack vf, vr, vd, D, bp, q= PolFun;
    b = ext.bgrid
    y = ext.ygrid
    ydef = ext.ydef
    p0 = findmin(abs.(0 .- b))[2]
    # --------------------------------------------------------------
    # 1. Data for figures
    # 1.1 Bond Prices
    b_in = findall(x -> x >= -0.30 && x <= 0, b)
    y_li = findmin(abs.(y .- 0.95 * mean(y)))[2]
    y_hi = findmin(abs.(y .- 1.05 * mean(y)))[2]
    q_gra = [q[b_in, y_li] q[b_in, y_hi]]
    # 1.2 Savings Policies
    b_in2 = findall(x -> x >= -0.3 && x <= 0.2, b)
    b_p = [bp[b_in2, y_li] bp[b_in2, y_hi]]
    # 1.3 Value function
    vf_p = [vf[b_in2, y_li] vf[b_in2, y_hi]]
    # --------------------------------------------------------------
    # 2. Figures
    if ~isdir(".\\Figures")
        mkdir(".\\Figures")
    end
    # 2.1 Bond Price
    title = titles[1]
    plot(
        b[b_in],
        q_gra,
        xlabel = "B'",
        lw = [1.15],
        title = "Bond price schedule q(B',y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
    # 2.2 Savings Policies
    title = titles[2]
    plot(
        b[b_in2],
        b_p,
        xlabel = "B",
        lw = [1.15],
        title = "Savings function B'(B,y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
    # 2.3 Value function
    title = titles[3]
    plot(
        b[b_in2],
        vf_p,
        xlabel = "B",
        lw = [1.15],
        title = "Value function vo(B,y)",
        titlefont = font(12),
        linestyle = [:solid :dash],
        linecolor = [:red :blue],
        label = ["y-low." "y-high"],
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legendfontsize = 8,
        legend = :topleft,
    )
    savefig(".\\Figures\\$title")
end

# [2]	Graphics for Simulation
function graph_simul(
    Sim;
    smpl = 1:500,
    titles = ["FigSim1.svg" "FigSim2.svg" "FigSim3.svg"],
    )
    default = Sim[smpl, 5]

    p1 = plot(Sim[smpl, 3], label = "", title = "Output (t)", lw = 2)
    p1 = bar!(twinx(),default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p2 = plot(Sim[smpl, 4], label = "", title = "Issued Debt (t+1)", lw = 2)
    p2 = bar!(twinx(),default, fc = :grey,lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p3 = plot(Sim[smpl, 4] ./ Sim[smpl, 3], label = "", lw = 2,
        title = "Debt to output ratio (t)",
    )
    p3 = bar!(twinx(), default,fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p4 = plot(Sim[smpl, 6], label = "", title = "Value function (t)", lw = 2)
    p4 = bar!(twinx(), default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    p5 = plot(Sim[smpl, 7], label = "", title = "Bond price q(t)", lw = 2)
    p5 = bar!(twinx(), default, fc = :grey, lc = :grey, label = "",
        alpha = 0.15, yticks = nothing, framestyle = :none,
    )
    if ~isdir(".\\Figures")
        mkdir(".\\Figures")
    end
    title = titles[1]
    plot(p1, p2, layout = (2, 1))
    savefig(".\\Figures\\$title")
    title = titles[2]
    plot(p4, p5, layout = (2, 1))
    savefig(".\\Figures\\$title")
    title = titles[3]
    plot(p3)
    savefig(".\\Figures\\$title")
    display("See your graphics")
end


###############################################################################
# [-] ADDITIONAL SUPPORTING CODES
###############################################################################

function mytauch(μ::Float64, ρ::Float64, σrisk::Float64, N::Int64, m::Int64)
    #= -----------------------------------------------------
    Tauchen discretization
    ----------------------------------------------------- =#
    if (N % 2 != 1)
        return "N should be an odd number"
    end
    grid = (2 * m) / (N - 1)
    sig_z = σrisk / sqrt(1 - ρ^2)
    Z = -m:grid:m
    Z = μ .+ Z .* sig_z
    d5 = 0.5 * (Z[2] - Z[1])
    pix = Array{Float64,2}(undef, N, N)
    for i = 1:N
        s = -(1 - ρ) * μ - ρ * Z[i]
        pix[i, 1] = cdf(Normal(), (Z[1] + d5 + s) / σrisk)
        for j = 2:N-1
            pix[i, j] =
                cdf(Normal(), (Z[j] + d5 + s) / σrisk) -
                cdf(Normal(), (Z[j] - d5 + s) / σrisk)
        end
        pix[i, N] = 1 - cdf(Normal(), (Z[N] - d5 + s) / σrisk)
    end
    pix = pix ./ sum(pix, dims = 2) # Normalization to achieve 1
    return Z, pix
end
