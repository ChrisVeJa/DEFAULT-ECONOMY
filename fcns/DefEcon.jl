#= #############################################################################
 			APPROXIMATING DEFAULT ECONOMIES WITH NEURAL NETWORKS
The following module has all the main functions used in the master.jl These are:

    1. SolveR: It solves a Default Economy, its output are:
        * ModelSolve: A NamedTuple with the following fields:
            ** Mod : A namedtuple with the parameters Θ, the h() function
                for default, and the utility function uf()
            ** PolFun: A NamedTuple with the policy functions and the values
                functions. Its components are:
                - VF: value function
                - VC: Value of continuation under no default
                - VD: valuye of default
                - D: Default choices
                - BP: Policy function for Bₜ₊₁
                - Price: qₜ₊₁
            ** Ext: SOme extra feautures of the model as the grids for bonds
                outputs, the output level under default, and the markovmatrix P
    2. ModelSim: It simulates the economy, the output is modelsim which has two
        fields: a) Sim, a matrix with the simulation, and b) order,a  string
        matrix with the name of the columns in Sim.

    3. NeuTra : It trains a neural network with softplus activation function,
        returning:
            * DataNorm: The normalized input data
            * Data: The raw data
            * Hat: The predicted
            * Mhat: Neural Network structure in a chain-type format

Written by: Christian Velasquez (velasqcb@bc.edu)
Dont hesitate in send any comment
############################################################################# =#

module DefEcon
using Random,
    Distributions, Statistics, LinearAlgebra, Plots, StatsBase, Parameters, Flux
include("supcodes.jl")
include("mainsup.jl")
include("graphs.jl")
# ==============================================================================
# [1] SOLUTION OF THE MODEL
# ==============================================================================
function Solver(params, deffun, utifun)
    #=
        This function solvesa standard Default with economy as in Arellano 2008
        The steps to solve it are:
            [1] Find the discret points from the tauchen algorithm
            [2] Calculates h(y) and u(h(y)) where h() is the function for
                out in default case
            [3] Calculate the grid for the support of `b`
            [4] Solve the fixed point problem (see respective code)
    =#

    # --------------------------------------------------------------
    # 0. Unpacking Parameters
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    # --------------------------------------------------------------
    # 1. Tauchen discretization of log-output
    ly, P = mytauch(μ, ρ, η, nx, m)
    y = exp.(ly)
    # --------------------------------------------------------------
    # 2. Output in case of default
    ydef = deffun(y, fhat)
    udef = utifun.(ydef, σrisk)
    # --------------------------------------------------------------
    # 3. To calculate the intervals of debt I will consider
    grid = (ub - lb) / (ne - 1)
    b = [lb + (i - 1) * grid for i = 1:ne]
    p0 = findmin(abs.(0 .- b))[2]
    b[p0] = 0
    # --------------------------------------------------------------
    # 4. Solving the fixed point problem
    vf, vr, vd, D, bp, q = FixedPoint(b, y, udef, P, p0, params, utifun)
    PolFun = (vf = vf, vr = vr, vd = vd, D = D, bp = bp, q = q)
    ModelSolve = (
        Mod = (Θ = params, h = deffun, uf = utifun),
        PolFun = PolFun,
        ext = (bgrid = b, ygrid = y, ydef = ydef, P = P),
    )
    display("Model solved, please see your results")
    return ModelSolve
end

# ==============================================================================
# [2] SIMULATING THE ECONOMY
# ==============================================================================
function ModelSim(params, PolFun, ext; nsim = 100000, burn = 0.05)
    #=
        This function is setting the initial features to simulate the Economy
        Its structure is as follow
            [1] Define some basics value for the simulation
            [2] Simulates the path of states for output conditional
                on the markov matrix P, if nseed is different to 0 a seed is
                put in order to have the same solution always.
                    ``nseed`` should be a vector with two entrances
            [3] The initial point is a economy not in default with a
                debt level of 0

    =#
    # -------------------------------------------------------------------------
    # 0. Settings
    @unpack r, σrisk, ρ, η, β, θ, nx, m, μ, fhat, ne, ub, lb, tol = params
    P = ext.P
    b = ext.bgrid
    y = ext.ygrid
    ydef = ext.ydef
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
# ==============================================================================
end
