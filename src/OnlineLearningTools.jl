module OnlineLearningTools

using Enzyme: Reverse, Annotation, Active, Const, DuplicatedNoNeed, autodiff, set_runtime_activity

using Enzyme.EnzymeRules: RevConfig, AugmentedReturn
import Enzyme.EnzymeRules: reverse, augmented_primal

"""
    c = rcopy(x)

Return a copy of `x` that will not be modified if `x` is updated after calling `rcopy`. 
Acts recursively on (named) tuples.
`rcopy` is used to store the history of the `state` argument of `repeat` 
and may need to be specialized for user-defined types.
"""
rcopy(x::AbstractArray) = 1*x    # copy(x) triggers a strange Enzyme bug !!
rcopy(x::Union{<:Tuple,<:NamedTuple}) = map(rcopy, x)

"""
    new_scratch = rsimilar(scratch)

Allocates a scratch space similar to `scratch`. 
Acts recursively on (named) tuples, and calls `Base.similar` otherwise.
`rsimilar` is used on argument `scratch` of `online_loss` 
and may need to be specialized for user-defined types.
"""
rsimilar(x) = similar(x)
rsimilar(x::Union{<:Tuple, <:NamedTuple}) = map(rsimilar, x)

include("repeat.jl")
include("online_loss.jl")

end
