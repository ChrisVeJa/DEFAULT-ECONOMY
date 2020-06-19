using Juno
import Zygote: Params, gradient
"""
  update!(x, x̄)
Update the array `x` according to `x .-= x̄`.
"""
function update!(x::AbstractArray, x̄)
  x .-= x̄
end

"""
    update!(opt, p, g)
    update!(opt, ps::Params, gs)
Perform an update step of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).
As a result, the parameters are mutated and the optimizer's internal state may change.
"""
