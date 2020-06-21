1. Defining the initial point (y=1, B=0, D=0). This corresponds to an economy
   with zero debt with acces to the market and in its deterministic average
   output level.

2. Conditional in the transition matrix P(y'|y), I simulate 105000 future realizations
   of the output and create a container matrix [Dₜ₋₁,Bₜ, yₜ, Bₜ₊₁, Dₜ, Vₜ, qₜ(bₜ₊₁(bₜ,yₜ))]

3. If the country is not in default, it chooses Dₜ, and Bₜ₊₁ according to the policy
   functions previously computed, leading a new price qₜ₊₁.

4. If the country choose to default in the previous period, I simulate a realization
   of the re-entering state  ReEnter ∼ Bern(θ). If the country is accepted to re-enter
   it chooses as in point 3, otherwise it stays in the default state.

5. The pseudo code is:
```
    if Dₜ₋₁ == 0
        Dₜ  = Dₜ(B,y)
        yₜ  = (1-D) yᴺᴰ + D* yᵈᵉᶠ
        B' = (1-D) * B'(B,y)
    else
        Draw ReEnter ∼ Bern(θ)
        if ReEnter = 1
            Dₜ  = Dₜ(B,y)
            yₜ  = (1-D) yᴺᴰ + D* yᵈᵉᶠ
            B' = (1-D) * B'(B,y)
        else
            yₜ  = yᵈᵉᶠ
            Bₜ  = 0
        end
    end
```
6. I discard the first 5000 elements to avoid any effect of the intial point
