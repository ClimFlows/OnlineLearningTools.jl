using OnlineLearningTools: repeat, do_repeat
using Enzyme
using Test

macro twice(expr)
    return esc(quote
                   @showtime $expr
                   @showtime $expr
                   @test true
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

function L2(x, y) # = mapreduce(diff2, +, x,y)
    ret = zero(eltype(x))
    for i in eachindex(x, y)
        @inbounds ret += (x[i] - y[i])^2
    end
    return ret
end
# L2(x, y) = sum((x - y) .^ 2)

diff2(x, y) = (x - y)^2
diff2(x::V, y::V) where {V<:Vector} = @inline mapreduce(diff2, +, x, y)

dup(x) = Duplicated(x, make_zero(x))

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

    function grad(loss, state)
        dstate = make_zero(state)
        dparams = make_zero(params)
        autodiff(Reverse, loss, Active, dup(target), Duplicated(state, dstate),
                 dup(scratch), Duplicated(params, dparams))
        return dstate, dparams
    end

    repeat(fun!, nstep, target, scratch, params)

    state0 = randn(n)
    @twice loss_auto(target, copy(state0), scratch, params)
    @twice grads_auto = gradient(Reverse, loss_auto, target, copy(state0), scratch, params)
    #    @twice grad(loss_auto, copy(state0))

    #    @twice autodiff(Reverse, repeat, Const, Const(fun!), Const(nstep), dup(copy(state0)), dup(scratch), dup(params))
    @twice loss_custom(target, copy(state0), scratch, params)
    #    @twice grad(loss_custom, copy(state0))
    @twice grads_custom = gradient(Reverse, loss_custom, target, copy(state0), scratch,
                                   params)

    @test grads_auto[4] ≈ grads_custom[4]
    @test grads_auto[2] ≈ grads_custom[2]
    @test grads_auto[1] ≈ grads_custom[1]
end

@testset "OnlineLearningTools.jl" begin
    test_repeat()
end
