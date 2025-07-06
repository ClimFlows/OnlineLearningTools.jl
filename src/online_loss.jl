"""
    online_loss = OnlineLoss(fun!, n, state, scratch, loss)
    l = online_loss(params)

Equvalent to:

    l = 0
    for i in 1:n
        fun!(i-1, state, scratch, params) # advance from step i-1 to step i
        l += loss(i, state)
    end

The following assumptions are made:
+ `fun!` may be a closure but it does not contain writable buffers
+ `state` is both read and modified by `fun!`
+ `scratch` may be written to by fun! but values passed to `fun!` are ignored
+ `loss` is a pure function or closure: it does not modify its inputs or captured variables
+ `params` are read but not modified by `fun!`

`(::OnlineLoss)(params)` has a custom Enzyme adjoint, whose correctness relies on these assumptions.
"""
struct OnlineLoss{Fun!, Scratch, State, Loss}
    fun!::Fun!
    Nstep::Int
    initial::State
    scratch::Scratch
    loss::Loss
end

(loss::OnlineLoss)(params) = online_loss(loss, params)

function online_loss(loss, params)
    state = rcopy(loss.initial)
    scratch = rsimilar(loss.scratch)
    l = loss.loss(0, state)
    for i in 1:loss.Nstep
        loss.fun!(i-1, state, scratch, params) # advance state from i-1 to i
        l += loss.loss(i, state)
    end
    return l
end

#=
function augmented_primal(::RevConfig,
                          loss_::Const{<:OnlineLoss},
                          ::Type{<:Active},
                          params::Annotation)
    loss = loss_.val
    state = rcopy(loss.initial)
    scratch = rsimilar(loss.scratch)
    states = Vector{typeof(state)}(undef, loss.Nstep)
    states[1] = rcopy(state)
    for i in 1:loss.Nstep
        loss.fun!(i-1, state, scratch, params.val) # advance state from i-1 to i
        states[i] = rcopy(state)
    end
    l = sum(loss.loss(i, tape[i+1]) for i in 0:Nstep)
    tape = (; states, scratch)
    return AugmentedReturn(l, nothing, tape )
end

function reverse(::RevConfig,
                loss_::Const{<:OnlineLoss},
                ::Type{<:Active},
                tape,
                params::Annotation)
    # Example with i = 0,1,2,3
    # State: ϕ    Params: θ
    # Aₙ : ∂ϕₙ₊₁ ↦ (∂ϕₙ, ∂θ)
    # (∂ϕ₃, ∂θ₃) = (∇L₃, ∂0)
    # (∂ϕ₂, ∂θ₂) = (∇L₂, ∂θ₃) + A₂⋅∂ϕ₃
    # (∂ϕ₁, ∂θ₁) = (∇L₁, ∂θ₂) + A₁⋅∂ϕ₂
    # (∂ϕ₀, ∂θ₀) = (∇L₀, ∂θ₁) + A₀⋅∂ϕ₁
    # 
    # Init
    # ∂ϕ'← ∇L₃
    # Loop
    # ∂ϕ ← ∇L₂
    # (∂ϕ, ∂θ) ← (∂ϕ, ∂θ) + A₂⋅∂ϕ'
    # ∂ϕ ↔ ∂ϕ'
    # Loop
    # ∂ϕ ← ∇L₁
    # (∂ϕ, ∂θ) ← (∂ϕ, ∂θ) + A₁⋅∂ϕ'
    # ∂ϕ ↔ ∂ϕ'
    # Finalize
    # ∂ϕ ← 0
    # (∂ϕ, ∂θ) ← (∂ϕ, ∂θ) + A₀⋅∂ϕ'

    loss = loss_.val
    (; states, scratch) = tape
    scratch = Duplicated(scratch, make_zero(scratch))
    # Init
    ∇L = make_zero(loss.initial)
    ∂ϕ = make_zero(loss.initial)
    autodiff(Reverse, loss.loss, Active, Const(i), DuplicatedNoNeed(states[end], ∂ϕ))
    # Loop
    for i in loss.Nstep:-1:1
        autodiff(Reverse, loss.loss, Active, Const(i), Duplicated(states[i], ∂ϕ))                         # ∂ϕ ← ∇Lᵢ₋₁
        ∇L, ∂ϕ = ∂ϕ, ∇L                                                                                   # ∂ϕ ↔ ∂ϕ'
        autodiff(set_runtime_activity(Reverse), fun!, Const, Duplicated(states[i], ∂ϕ), scratch, params)  # (∂ϕ, ∂θ) ← (∂ϕ, ∂θ) + Aᵢ₋₁⋅∂ϕ'
    end
end

=#