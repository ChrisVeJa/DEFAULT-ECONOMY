##  APPROXIMATION OF DEFAULT ECONOMIES WITH NEURAL NETWORKS
### 1. Setting the model:

This document is based on the basic model from Arellano (2008), in this economy the country is populated by a representative agent who maximizes the expected value of her consumption path which can be smoothed by issued sovereign debt traded in a competitive foreign market compounded by atomistic risk neutral investors.

This debt is defaultable but if the country chooses to do it, he faces an instantaneous output cost and a exclusion from financial markets, In addition, there is a probability θ of re-entering next period. Hence, the houshehold problem is summarized as:
```math
    max V₀ = ∑ βᵗU(cₜ)
    s.t.
        cₜ = Dₜ * yₜᵈᵉᶠ + (1-Dₜ)(yₜ + Bₜ - qₜ₊₁(Bₜ₊₁,yₜ)*Bₜ₊₁)
```
where ```Dₜ``` is the default decision (`D = 1` if country defaults), ``yₜᵈᵉᶠ = h(yₜ)`` is the output level under default, with
```math
    h(yₜ) = min(y,γ* ̄y)
```

This problem can be expresed in terms of Value Functions following
```math
    vᵒ(Bₜ,yₜ) = max[ vᶜ(Bₜ,yₜ) , vᴰ(yₜ) ]
    s.t.
        vᶜ(Bₜ,yₜ) = max{U(yₜ + Bₜ - qₜ₊₁(Bₜ₊₁,yₜ)*Bₜ₊₁)+β ∫vᵒ(Bₜ₊₁,yₜ₊₁)f(y'|y)dy'}
        vᴰ(yₜ)   = u(yₜᵈᵉᶠ) + β∫[θvᶜ(0, yₜ₊₁) + (1-θ)vᴰ(yₜ₊₁)]f(y'|y)dy
```
Let ``D(B)`` the default set, defining as:
```math
    D(B) = {y ∈ Y: vᶜ(B,y) < vᵈ(y)}
```
leading a probability of default: `` δ(B',y) = ∫f(y'|y)dy |D(B')``
To close the model, the equilibrium bond price is:
```math
    qₜ₊₁(Bₜ₊₁,yₜ) = (1-δ)/(1+r)
```
### 2. Recursive Equilibrium description
In this economy the equilibrium is a list of policy functions for consumption, debt, default a repayments sets, and bond prices such that:

    1. Taking as given everything else consumption maximize utility and satisfy the resource constraint

    2. Taking as given the bond price; the policy function for debt B0 and sets A(B) , D(B) are in line with the government maximization problem

    3. Bond prices are consistent with the default set and the zero profit condition.

### 3. Setting and solving the model
A clue to solve this model is to know that when ``B = 0`` the country is indifferent between defaulting or not, then ``vᵒ = vᶜ = vᵈ``.

Now, it is time to solve the model in the computer!!!
First we include our module for this kind of economy
```julia
using Random, Distributions,Statistics, LinearAlgebra, Plots,StatsBase,Parameters, Flux;
include("DefaultEconomy.jl");
```

Following Arellano (2008), we calibrate the parameter as:

| Parameter     | Description                               | Value     |
| ----------    | -----------                               | -----     |
| `β`           | Discount factor                           | 0.953     |
| `r`           | Risk free interest rate                   | 0.017     |
| `σ`           | Risk aversion parameter                   | 2.0       |
| `θ`           | Probability of re-enter after a default   | 0.282     |
| `γ` (fhat)    | Cost factor of default                    | 0.969     |
| `Nₑ`          | Number of possibles debt levels           | 251       |
| `ub`          | Upper bound for assets position           | 0.4       |
| `lb`          | Lower bound for assets position           | -0.4      |
| `ρ`           | Persitance of the log-output              | 0.945     |
| `η`           | Variance of the log-output shock          | 0.025     |
| `μ`           | Mean of the output shock                  | 0         |
| `Nₓ`          | Number of points to approximate `y`       | 21        |
| `m`           | Number of s.d. covers by Tauchen          | 3         |

Additionally, we incorporate two parameters `tol=1e-8` as the convergence acceptance level and `maxite=1000` as the maximum number of iterations. The function `ModelSettings()` creates a structure of type `ModelSttings` with this characteristics
```julia
    EconDef = DefaultEconomy.ModelSettings();
```
Hence, `EconDef` is a type `ModelSettings` with the fields:
```julia
 Params::NamedTuple;    # Parameters
UtilFun::Function;      # Utility Function
   yDef::Function;      # Default Loss
```
If we prefer we can change the parameters of the model changed directly `EconDef` given that it is mutable. Once we define the settings of the model we find the solution typing
```julia
EconSol = DefaultEconomy.SolveDefEcon(EconDef);
```
The function `SolveDefEcon` takes a ModelSettings struct as argument and solve the economy using value function iteration, giving as output a type `ModelSolve` compounded by the characteristics:
```julia
Settings::ModelSettings;		# Settings of the Model
Solution::PolicyFunction;		# Solution of the Model
```
`Solution` itself is a structure whith
```julia
 valuefunction::Array{Float64};  # Value Function
valueNoDefault::Array{Float64};  # Value Function of no defaulting
  valueDefault::Array{Float64};  # Value Function of default
  defaulchoice::Array{Float64};  # Default Choice (1:= default)
    bondpolfun::Array{Float64};	 # Issued Debt for t+1
     bondprice::Array{Float64};	 # Price of the debt
```
By default there is a graphic method that we can call by typing
```julia
DefaultEconomy.graph_solve(EconSol);
```
which generates similar figures to the original work

![imagen1](.//Figures//ValFun.png)
![imagen2](.//Figures//Savings.png)
![imagen3](.//Figures//BondPrice.png)

**Note**: The levels `low` and `high` are calculated as a 5% deviation respect the mean `y`

### 4. Simulating the model
