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
    @epochs N body
Run `body` `N` times. Mainly useful for quickly doing multiple epochs of
training in a REPL.
# Examples
```jldoctest
julia> Flux.@epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello
```
"""
