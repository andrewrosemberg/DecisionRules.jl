import QuickPOMDPs: QuickPOMDP
import POMDPTools: ImplicitDistribution
import Distributions: Normal
using POMDPs
using Flux
using Crux

mountaincar = QuickPOMDP(
    actions = [-1., 0., 1.],
    obstype = Float64,
    discount = 0.95,

    transition = function (s, a)        
        ImplicitDistribution() do rng
            x, v = s
            vp = v + a*0.001 + cos(3*x)*-0.0025 + 0.0002*randn(rng)
            vp = clamp(vp, -0.07, 0.07)
            xp = x + vp
            return (xp, vp)
        end
    end,

    observation = (a, sp) -> Normal(sp[1], 0.15),

    reward = function (s, a, sp)
        if sp[1] > 0.5
            return 100.0
        else
            return -1.0
        end
    end,

    initialstate = ImplicitDistribution(rng -> (-0.2*rand(rng), 0.0)),
    isterminal = s -> s[1] > 0.5
)

