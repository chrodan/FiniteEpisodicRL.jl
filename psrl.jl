type PSRL<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    Pcur::Array{Float64, 4}
    α::Float64 # Dirichlet prior for transitions
    τ::Float64 # Precision of Gaussian prior for mean returns
    rewards_known::Bool
end

function PSRL(S, A, H, rewards, α)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    Pcur = zeros(S, S, A, H)
    Pcur[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    m = PSRL(policy, ns, n, rewards, Pcur, α, 0., true)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
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
    R = m.rewards
    if !m.rewards_known
        error("Not implemented")
    end
    P, R
end

function update_policy!(m::PSRL)
    H = horizon(m)
    S = nS(m)
    A = nA(m)
    P, R = sample_MDP!(m)
    Vold = zeros(S)
    Qnew = zeros(S, A)
    
    for t=H:-1:1
        Qnew = R[:, :, t] + squeeze(sum(Vold .* P[:, :, :, t], 1), 1)
        for s=1:S
            m.policy[s, t] = indmax(Qnew[s, :])
            Vold[s] = Qnew[s, m.policy[s, t]]
        end
    end
end
