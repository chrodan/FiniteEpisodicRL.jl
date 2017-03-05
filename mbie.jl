type MBIE<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    Vopt::Matrix{Float64}
    δ::Float64
    ϵ::Float64
    rewards_known::Bool
    maxobs::Float64
    maxRet::Float64
    maxR::Float64
    keep_updating::Bool
end

function MBIE(S, A, H, rewards, δ, ϵ, keep_updating=false)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    maxRet = sum(maximum(reshape(rewards, S*A, H), 1))
    # use asymptotic form here, may be off by a constant factor
    maxobs = 4 * S / ϵ^2 * H^4 + 1 / ϵ^2 * H^4 * log(S * A * H / ϵ / δ)
    
    m = MBIE(policy, ns, phat, n, rewards, zeros(S, H+1), δ, ϵ, true, maxobs, maxRet, maximum(rewards[:]), keep_updating)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end
function observe!(m::MBIE, s, a, t, r, sn) 
    if m.keep_updating || m.n[s,a,t] < m.maxobs
        m.ns[sn, s, a, t] += 1
        m.n[s, a, t] +=  1
        if !m.rewards_known
            m.rewards[s, a, t] += r
        end
    end
end
maxV(m::MBIE) = m.maxRet
maxR(m::MBIE) = m.maxR
function update_policy!(m::MBIE)
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    δ = m.δ
    ϵ = m.ϵ
    Vmax = maxV(m)
    Rmax = maxR(m)
    Q = zeros(A)

    # precomputations for confidence intervals
    m_val = m.maxobs
    lR = log(4 * S * A * m_val / δ) / 2
    lC = if S < 30
        log(2^S - 2) + log(2 * S * A * m_val / δ)
    else
        S*log(2) + log(2 * S * A * m_val / δ)
    end
    m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    for t=H:-1:1
        V = m.Vopt[:, t+1]
        ind = sortperm(V)
        curmaxV = min(V[ind[end]], Vmax)

        for s ∈ 1:S
            for a ∈ 1:A
                slack = √(2 * lC / m.n[s, a, t])
                mass_left = 1.
                EV = 0.
                for i ∈ S:-1:1
                    ii = ind[i]
                    ph = m.phat[ii, s, a, t]
                    pmax = min(mass_left, ph + slack)
                    slack -= pmax - ph
                    mass_left -= pmax
                    EV += pmax * V[ii]
                    if mass_left < 3 * eps(typeof(mass_left))
                        break
                    end
                end
                                
                r = m.rewards[s, a, t]
                if !m.rewards_known
                    min(m.rewards[s, a, t] / n[s, a, t] + Rmax * √(lR / m.n[s, a, t]), Rmax)
                end
                
                Q[a] = r + EV
            end
            bestA = indmax(Q)
            m.policy[s, t] = bestA
            m.Vopt[s, t] = Q[bestA]
        end
    end
end
