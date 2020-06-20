##  APPROXIMATION OF DEFAULT ECONOMIES WITH NEURAL NETWORKS
### 1. Setting the model:

This document is based on the basic model from Arellano (2008), in this economy the country is populated by a representative agent who maximizes the expected value of her consumption path which can be smoothed by issued sovereign debt traded in a competitive foreign market compounded by atomistic risk neutral investors.

This debt is defaultable but if the country chooses to do it, he faces an instantaneous output cost and a exclusion from financial markets, In addition, there is a probability θ of re-entering next period. Hence, the houshehold problem is summarized as:
```math
    max V₀ = ∑ βᵗU(cₜ)
    s.t.
        cₜ = Dₜ * yₜᵈᵉᶠ + (1-Dₜ)(yₜ + Bₜ - qₜ₊₁(Bₜ₊₁,yₜ)*Bₜ₊₁)
```
where ```Dₜ``` is the default decision (`D = 1` if country defaults), ``yₜᵈᵉᶠ = h(yₜ)`` is the output level under default.

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
    qₜ₊₁(Bₜ₊₁,yₜ) = (1-δ)/ (1+r)
```
### 2. Recursive Equiulibrium description
In this economy the equilibrium is a list of policy functions for consumption, debt, default a repayments sets, and bond prices such that:

    1. Taking as given everything else consumption maximize utility and satisfy the resource constraint

    2. Taking as given the bond price; the policy function for debt B0 and sets A(B) , D(B) are in line with the government maximization problem

    3. Bond prices are consistent with the default set and the zero profit condition.

A clue to solve this model is to know that when ``B = 0`` the country is indifferent between defaulting or not, then ``vᵒ = vᶜ = vᵈ``.

Now, it is time to solve the model in the computer!!!

```julia
  f(x)  = log(x)
  f(x,y) = log(x)-y + θ(2)t₁
```


```math
 x = f(x) + t₁ +
```
