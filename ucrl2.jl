"""Version of UCRL2 adapted to finite horizon
Confidence intervals taken from Ian Osband's TabulaRL
package"""
type UCRL2<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    nstemp::Array{UInt64, 4}
    ntemp::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    rewardstemp::Array{Float64, 3}
    Vopt::Matrix{Float64}
    δ::Float64
    rewards_known::Bool
    maxRet::Float64
    maxR::Float64
    immediate_updates::Bool
    update_necessary::Bool
end

function UCRL2(S, A, H, rewards, δ, immediate_updates=false)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    nstemp = zeros(UInt64, S, S, A, H)
    ntemp = ones(UInt64, S, A, H)
    rewardstemp = zeros(S, A, H)
    maxRet = sum(maximum(reshape(rewards, S*A, H), 1))
    m = UCRL2(policy, ns, phat, n, nstemp, ntemp, rewards, rewardstemp, zeros(S, H+1), δ, true, maxRet, maximum(rewards[:]), immediate_updates, false)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function observe!(m::UCRL2, s, a, t, r, sn) 
    m.nstemp[sn, s, a, t] += 1
    m.ntemp[s, a, t] +=  1
    m.rewardstemp[s, a, t] += r
    if m.immediate_updates || m.ntemp[s,a,t] >= max(1, m.n[s,a,t])
        m.update_necessary = true
    end
end
maxV(m::UCRL2) = m.maxRet
maxR(m::UCRL2) = m.maxR

function update_policy!(m::UCRL2, force=false)
    if !force && !m.update_necessary
        return
    end

    # First make all counts up-to-date
    m.n += m.ntemp
    m.ns += m.nstemp
    if !m.rewards_known
        m.rewards += m.rewardstemp
    end
    fill!(m.ntemp, 0)
    fill!(m.nstemp, 0)
    fill!(m.rewardstemp, 0.)
    m.update_necessary=false

    S = nS(m)
    H = horizon(m)
    A = nA(m)
    δ = m.δ
    Vmax = maxV(m)
    Rmax = maxR(m)
    Q = zeros(A)
    k = sum(m.n[:,:, 1]) # number of observed episodes
    # precomputations for confidence intervals
    lR = 4 * log(4 * S * H * A * (k+1) / δ) 
    lC = 4 * S * log(2 * S * H * A * (k+1) / δ)

    m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    # Extended value iteration
    for t=H:-1:1
        V = m.Vopt[:, t+1]
        ind = sortperm(V)
        curmaxV = min(V[ind[end]], Vmax)

        for s ∈ 1:S
            for a ∈ 1:A
                slack = √(lC / m.n[s, a, t])
                mass_left = 1.
                EV = 0.
                for i ∈ S:-1:1
                    ii = ind[i]
                    ph = m.phat[ii, s, a, t]
                    pmax = min(mass_left, ph + slack)
                    slack -= pmax - ph
                    mass_left -= pmax
                    EV += pmax * V[ii]
                    if mass_left < 4 * eps(typeof(mass_left))
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
