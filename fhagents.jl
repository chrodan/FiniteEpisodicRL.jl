abstract FiniteHorizonAgent
nS(m::FiniteHorizonAgent) = size(m.policy, 1)
horizon(m::FiniteHorizonAgent) = size(m.policy, 2)
nA(m::FiniteHorizonAgent) = size(m.rewards, 2)

sample_action(m::FiniteHorizonAgent, s, t) = m.policy[s, t]
function observe!(m::FiniteHorizonAgent, s, a, t, r, sn) 
    m.ns[sn, s, a, t] += 1
    m.n[s, a, t] +=  1
    if !m.rewards_known
        m.rewards[s, a, t] += r
    end
end

include("psrl.jl")
include("ubev.jl")
include("mbie.jl")
include("mormax.jl")
include("ucrl2.jl")
include("medianpac.jl")
include("delayedql.jl")
include("oim.jl")
include("ucfh.jl")

