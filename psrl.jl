type PSRL<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    reward_precisions::Array{Float64, 3}
    Pcur::Array{Float64, 4}
    Rcur::Array{Float64, 3}
    α::Float64 # Dirichlet prior for transitions
    τ::Float64 # Precision of Gaussian prior for mean returns
    rewards_known::Bool
    Vopt::Matrix{Float64}
    Qopt::Array{Float64, 3}
end
function PSRL(S, A, H, α::Real, τ0::Real, τ::Real)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    Pcur = zeros(S, S, A, H)
    Pcur[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    Rcur = zeros(S, A, H)
    m = PSRL(policy, ns, n, zeros(S, A, H), fill(τ0, S, A, H), Pcur, Rcur, α, τ, false, zeros(S, H+1), zeros(S, A, H+1))
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end
function PSRL(S, A, H, rewards::Array, α)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    Pcur = zeros(S, S, A, H)
    Pcur[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    m = PSRL(policy, ns, n, rewards, zeros(S, A, H), Pcur, rewards, α, 0., true, zeros(S, H+1), zeros(S, A, H+1))
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function observe!(m::PSRL, s, a, t, r, sn) 
    m.ns[sn, s, a, t] += 1
    m.n[s, a, t] +=  1
    if !m.rewards_known
        τ1 = m.τ + m.reward_precisions[s,a,t]
        m.rewards[s, a, t] = (m.reward_precisions[s,a,t] * m.rewards[s,a,t] + m.τ * r) / τ1
        m.reward_precisions[s,a,t] = τ1
    end
end

function sample_MDP!(m::PSRL)
    S = nS(m)
    A = nA(m)
    H = horizon(m)
    P = m.Pcur
    αs = fill(m.α, S)
    for s=1:S, a=1:A, t=1:H
        αs .= m.ns[:, s, a, t] + m.α
        P[:, s, a, t] = rand(Dirichlet(αs))
    end

    R = m.Rcur
    if !m.rewards_known
        for s=1:S, a=1:A, t=1:H
            R[s,a,t] = randn() * 1 / √(m.reward_precisions[s,a,t]) + m.rewards[s,a,t] 
        end
    else
        R = m.rewards
    end
    P, R
end

function update_policy!(m::PSRL)
    H = horizon(m)
    S = nS(m)
    A = nA(m)
    P, R = sample_MDP!(m)
    Qnew = zeros(S, A)
    
    for t=H:-1:1
        Vold = m.Vopt[:, t+1]
        Qnew = R[:, :, t] + squeeze(sum(Vold .* P[:, :, :, t], 1), 1)
        for s=1:S
            m.policy[s, t] = indmax(Qnew[s, :])
            m.Vopt[s, t] = Qnew[s, m.policy[s, t]]
        end
        m.Qopt[:,:,t] = Qnew
    end
end
