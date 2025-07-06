"""
    repeat(fun!, n, state, scratch, params)

Equvalent to:

    for i in 1:n
        fun!(state, scratch, params)
    end
    return nothing

The following assumptions are made:
    - `fun!` may be a closure but it does not contain writable buffers
    - `state` is both read and modified by `fun!`
    - `scratch` may be written to by fun! but values passed to `fun!` are ignored
    - `params` are read but not modified by `fun!`
    - function `rcopy(state)` is implemented correctly.

`repeat` has a custom Enzyme adjoint, whose correctness relies on these assumptions.
"""
repeat(fun!, n, state, scratch, params) = do_repeat(fun!, n, state, scratch, params)

# for testing purposes
function do_repeat(fun!, n, state, scratch, params)
    for _ in 1:n
        fun!(state, scratch, params) # modifies state
    end
    return nothing
end

function augmented_primal(::RevConfig,
                          ::Const{typeof(repeat)},
                          ::Type{<:Annotation},
                          fun!::Annotation,
                          n::Annotation,
                          state::Annotation,
                          scratch::Annotation,
                          params::Annotation)
    st = state.val
    tape = Vector{typeof(st)}(undef, n.val)
    for i in 1:(n.val)
        tape[i] = rcopy(st)
        fun!.val(st, scratch.val, params.val) # modifies st
    end
    return AugmentedReturn(nothing, nothing, tape)
end

function reverse(::RevConfig,
                 ::Const{typeof(repeat)},
                 ::Type{<:Annotation},
                 tape,
                 fun!::Annotation,
                 n::Annotation,
                 state::Annotation,
                 scratch::Annotation,
                 params::Annotation)
    dstate = state.dval
    for i in (n.val):-1:1
#        autodiff(set_runtime_activity(Reverse), Const(fun!), Const, DuplicatedNoNeed(tape[i], dstate), scratch, params)
        autodiff(set_runtime_activity(Reverse), fun!, Const, DuplicatedNoNeed(tape[i], dstate), scratch, params)
    end
    return nothing, nothing, nothing, nothing, nothing
end
