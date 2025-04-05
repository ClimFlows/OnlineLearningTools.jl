using OnlineLearningTools: repeat, do_repeat
using Enzyme: gradient, Reverse
using Test

macro twice(expr)
    return esc(quote
                   @showtime $expr
                   @showtime $expr
               end)
end

# our discrete-time dynamical system
# we mostly want it to remain bounded when iterating many times
# so we just let params be random numbers ∈ [0,1]
function fun!(state, params, scratch)
    @. scratch = params * state
    @. state = scratch
    return nothing # required !
end

L2(x, y) = sum((x - y) .^ 2)

function test_repeat(n=100, nstep=20, target=randn(n), params=rand(n),
                     scratch=zero(params))
    function loss_auto(target, state, scratch, params)
        do_repeat(fun!, nstep, state, scratch, params)
        return L2(state, target)
    end

    function loss_custom(target, state, scratch, params)
        repeat(fun!, nstep, state, scratch, params)
        return L2(state, target)
    end

    repeat(fun!, nstep, target, scratch, params)

    state0 = randn(n)
    @twice loss_auto(target, copy(state0), scratch, params)
    @twice grads_auto = gradient(Reverse, loss_auto, target, copy(state0), scratch, params)
    @twice loss_custom(target, copy(state0), scratch, params)
    @twice grads_custom = gradient(Reverse, loss_custom, target, copy(state0), scratch,
                                   params)
    @test grads_auto[1] ≈ grads_custom[1]
    @test grads_auto[2] ≈ grads_custom[2]
    # we don't care about the gradient w.r.t. scratch
    @test grads_auto[4] ≈ grads_custom[4]
end

@testset "OnlineLearningTools.jl" begin
    test_repeat()
end
