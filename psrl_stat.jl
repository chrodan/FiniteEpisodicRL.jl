using Distributions

type PSRLStat<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 3}
    n::Array{UInt64, 2}
    rewards::Array{Float64, 2}
    reward_precisions::Array{Float64, 2}
    Pcur::Array{Float64, 3}
    Rcur::Array{Float64, 2}
    α::Float64 # Dirichlet prior for transitions
    τ::Float64 # Precision of Gaussian prior for mean returns
    rewards_known::Bool
    Vopt::Matrix{Float64}
    Qopt::Array{Float64, 3}
end
function PSRLStat(S, A, H, α::Real, τ0::Real, τ::Real)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A)
    Pcur = zeros(S, S, A)
    Pcur[1, :, :] = 1.
    n = zeros(UInt64, S, A)
    Rcur = zeros(S, A)
    m = PSRLStat(policy, ns, n, zeros(S, A), fill(τ0, S, A), Pcur, Rcur, α, τ, false, zeros(S, H+1), zeros(S, A, H+1))
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function observe!(m::PSRLStat, s, a, t, r, sn) 
    m.ns[sn, s, a] += 1
    m.n[s, a] +=  1
    if !m.rewards_known
        τ1 = m.τ + m.reward_precisions[s,a]
        m.rewards[s, a] = (m.reward_precisions[s,a] * m.rewards[s,a] + m.τ * r) / τ1
        m.reward_precisions[s,a] = τ1
    end
end

function sample_MDP!(m::PSRLStat)
    S = nS(m)
    A = nA(m)
    H = horizon(m)
    P = m.Pcur
    αs = fill(m.α, S)
    for s=1:S, a=1:A
        αs .= m.ns[:, s, a] + m.α
        P[:, s, a] = rand(Dirichlet(αs))
    end

    R = m.Rcur
    if !m.rewards_known
        for s=1:S, a=1:A
            R[s,a] = randn() * 1 / √(m.reward_precisions[s,a]) + m.rewards[s,a] 
        end
    else
        R = m.rewards
    end
    P, R
end

function update_policy!(m::PSRLStat)
    H = horizon(m)
    S = nS(m)
    A = nA(m)
    P, R = sample_MDP!(m)
    Qnew = zeros(S, A)
    
    for t=H:-1:1
        Vold = m.Vopt[:, t+1]
        Qnew = R + squeeze(sum(Vold .* P, 1), 1)
        for s=1:S
            m.policy[s, t] = indmax(Qnew[s, :])
            m.Vopt[s, t] = Qnew[s, m.policy[s, t]]
        end
        m.Qopt[:,:,t] = Qnew
    end
end
