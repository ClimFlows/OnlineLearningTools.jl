"""
    online_loss = OnlineLoss(fun!, n, state, scratch)
    l = online_loss(params)

Equvalent to:

    l = loss.fun!(1, state, scratch, params) # advance state from i=0 to i=1
    for i in 2:n
        l += loss.fun!(i, state, scratch, params) # advance state from i-1 to i
    end

The following assumptions are made:
+ `fun!` may be a closure but it does not contain writable buffers
+ `state` is both read and modified by `fun!`
+ `scratch` may be written to by fun! but values passed to `fun!` are ignored
+ `params` are read but not modified by `fun!`

`(::OnlineLoss)(params)` has a custom Enzyme adjoint, whose correctness relies on these assumptions.
"""
struct OnlineLoss{Fun!, Scratch, State}
    fun!::Fun!
    Nstep::Int
    initial::State
    scratch::Scratch
end

function (loss::OnlineLoss)(params) 
    state = rcopy(loss.initial)
    scratch = rsimilar(loss.scratch)
    return online_loss(loss, params, state, scratch)
end

# custom gradient

function online_loss(loss, params, state, scratch)
    l = loss.fun!(1, state, scratch, params) # advance state from i=0 to i=1
    for i in 2:loss.Nstep
        l += loss.fun!(i, state, scratch, params) # advance state from i-1 to i
    end
    return l
end

function augmented_primal(config::RevConfig,
                          ::Const{typeof(online_loss)}, ::Type{<:Active},
                          loss_::Annotation,
                          params_::Annotation,
                          state_::Annotation,
                          scratch_::Annotation)
    loss, params, state, scratch = loss_.val, params_.val, state_.val, scratch_.val
    states = Vector{typeof(state)}(undef, loss.Nstep)

    states[1] = rcopy(state)
    l = loss.fun!(1, state, scratch, params) # advance state from i=0 to i=1

    for i in 2:loss.Nstep
        states[i] = rcopy(state)
        l += loss.fun!(i, state, scratch, params) # advance state from i-1 to i
    end

    if needs_primal(config)
        result = l
    else
        result = nothing
    end
    return AugmentedReturn(result, nothing, states )
end

function reverse(::RevConfig,
                ::Const{typeof(online_loss)}, 
                dret::Active,
                states,
                loss_::Annotation,
                params::Annotation,
                state_::Annotation,
                scratch::Annotation)
    @assert dret.val==1
    loss, dstate = loss_.val, state_.dval
    for i in loss.Nstep:-1:1
        autodiff(set_runtime_activity(Reverse), Const(loss.fun!), Active, Const(i), Duplicated(states[i], dstate), scratch, params)
    end
    return nothing, nothing, nothing, nothing
end
