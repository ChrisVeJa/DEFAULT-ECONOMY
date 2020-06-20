##  APPROXIMATION OF DEFAULT ECONOMIES WITH NEURAL NETWORKS
### Setting the model:

This document is based on the basic model from Arellano (2008), in this economy the country is populated by a representative agent who maximizes the expected value of her consumption path which can be smoothed by issued sovereign debt traded in a competitive foreign market compounded by atomistic risk neutral investors.

This debt is defaultable but if the country chooses to do it, he faces an instantaneous output cost and a exclusion from financial markets, In addition, there is a probability θ of re-entering next period. Hence, the houshehold problem is summarized as:
[dada](https://latex.codecogs.com/gif.latex?E_0%5Cleft%5C%7B%20%5Csum_t%5E%7B%5Cinfty%7D%20%5Cbeta%5Et%20U%28c_t%29%5Cright%5C%7D)

```math
  max  V₀ = ∑ βᵗU(cₜ)
  s.t.
      cₜ = yₜᵈᵉᶠ
```
```julia
  f(x)  = log(x)
  f(x,y) = log(x)-y + θ(2)t₁
```


```math
 x = f(x) + t₁ +
```
