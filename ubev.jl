type UBEV<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    Vopt::Matrix{Float64}
    δ::Float64
    rewards_known::Bool
    maxRet::Float64
    maxR::Float64
    explore_bonus::Float64
end

function UBEV(S, A, H, rewards, δ, explore_bonus=1.)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    maxRet = sum(maximum(reshape(rewards, S*A, H), 1))
    m = UBEV(policy, ns, phat, n, rewards, zeros(S, H+1), δ, true, maxRet, maximum(rewards[:]), explore_bonus)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

maxV(m::UBEV) = m.maxRet
maxR(m::UBEV) = m.maxR

llnp(x) = max(0, log(max(0, log(max(0, x)))))
phi(n, S, A, H, δprime, range) = range * √(1./n * (2*llnp(n) + log(3 * S * A * H / δprime)))
confint(n, lC, valmax) =  valmax * √(1./n * (2*llnp(n) + lC))

function update_policy!(m::UBEV)
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    δprime = m.δ
    if m.rewards_known
        δprime /= 7
    else
        δprima /= 9
    end
    Vmax = maxV(m)
    Rmax = maxR(m)
    Q = zeros(A)
    lC = log(3 * S * A * H / δprime)
    m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    for t=H:-1:1
        V = m.Vopt[:, t+1]
        curmaxV = min(maximum(V), Vmax)
        curminV = min(minimum(V), Vmax)
        for s ∈ 1:S
            for a ∈ 1:A
                
                EV = curmaxV
                if m.n[s,a,t] > 0
                    EV = min(curmaxV, m.phat[:, s, a, t] ⋅ V + confint(m.n[s,a,t], lC, curmaxV - curminV) * m.explore_bonus) 
                end
                
                r = m.rewards[s, a, t]
                if !m.rewards_known
                    min(m.rewards[s, a, t] / n[s, a, t] + confint(m.n[s, a, t], lC, Rmax), Rmax)
                end
                
                Q[a] = r + EV
            end
            bestA = indmax(Q)
            m.policy[s, t] = bestA
            m.Vopt[s, t] = Q[bestA]
        end
    end
end
