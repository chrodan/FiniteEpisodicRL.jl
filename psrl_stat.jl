using Distributions
using StatsFuns
type PSRLStat<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    statemap::Vector{UInt64}
    effS::UInt64
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
    effS = 1
    statemap = zeros(UInt64, S)
    Pcur = zeros(S, S, A)
    Pcur[1, :, :] = 1.
    n = zeros(UInt64, S, A)
    Rcur = zeros(S, A)
    m = PSRLStat(policy, statemap, effS, ns, n, zeros(S, A), fill(τ0, S, A), Pcur, Rcur, α, τ, false, zeros(S, H+1), zeros(S, A, H+1))
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function observe!(m::PSRLStat, s, a, t, r, sn)
    if m.statemap[s] == 0
        m.statemap[s] = m.effS
        m.effS += 1     
    end
    si = m.statemap[s]
    if m.statemap[sn] == 0
        m.statemap[sn] = m.effS
        m.effS += 1     
    end
    sni = m.statemap[sn]

    m.ns[sni, si, a] += 1
    m.n[si, a] +=  1
    if !m.rewards_known
        τ1 = m.τ + m.reward_precisions[si,a]
        m.rewards[si, a] = (m.reward_precisions[si,a] * m.rewards[si,a] + m.τ * r) / τ1
        m.reward_precisions[si,a] = τ1
    end
end

function sample_MDP!(m::PSRLStat)
    S = nS(m)
    eS = S
    A = nA(m)
    H = horizon(m)
    P = m.Pcur
    Threads.@threads for i in eachindex(P, m.ns)
        P[i] = StatsFuns.RFunctions.gammarand(m.ns[i] + m.α, 1.)
    end
    P ./= sum(P, 1)
    R = m.Rcur
    if !m.rewards_known
        R[:,:] = randn(eS, A) ./ sqrt.(m.reward_precisions) .+ m.rewards 
    else
        R = m.rewards
    end
    P, R
end

function update_policy!(m::PSRLStat)
    H = horizon(m)
    S = nS(m)
    eS = S
    A = nA(m)
    P, R = sample_MDP!(m)
    Qnew = zeros(eS, A)
    
    for t=H:-1:1
        Vold = m.Vopt[1:eS, t+1]
        Qnew = R + squeeze(sum(Vold .* P, 1), 1)
        for s=1:eS
            m.policy[s, t] = indmax(Qnew[s, :])
            m.Vopt[s, t] = Qnew[s, m.policy[s, t]]
        end
        m.Qopt[1:eS,:,t] = Qnew
    end
end

function sample_action(m::PSRLStat, s, t)
    i = m.statemap[s]
    if i == 0
        rand(1:nA(m))
    else
        m.policy[i, t]
    end
end
