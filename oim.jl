
type OIM<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    optR::Vector{Float64}
    Ve::Matrix{Float64}
    Vr::Matrix{Float64}
    δ::Float64
    ϵ::Float64
    rewards_known::Bool
end

function OIM(S, A, H, rewards, δ, ϵ, keep_updating=false)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S+1, S, A, H)
    phat = zeros(S+1, S, A, H)
    # pretend we have seen one transition to optimistic state S+1 from each state
    n = ones(UInt64, S, A, H)
    ns[end, :, :, :] = 1
    max_rewards = vec(maximum(reshape(rewards, S*A, H), 1))
    # rough guess of required optimistic rewards, exact value will require transferring entire theory to our setting
    #optR = 8 * cumsum(max_rewards).^2 * H^2 / ϵ^2 * log(S * A * H / ϵ / δ)
    optR = cumsum(max_rewards).^2 * H  / ϵ * log(S * A * H / δ)
    
    m = OIM(policy, ns, phat, n, rewards, optR, zeros(S, H+1),zeros(S, H+1), δ, ϵ, true)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

function update_policy!(m::OIM)
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    
    Qe = zeros(A)
    Qr = zeros(A)

    m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    for t=H:-1:1
        for s ∈ 1:S
            for a ∈ 1:A
                Qe[a] = m.optR[t] * m.phat[S+1, s, a, t] + sum(m.phat[1:S, s, a, t] .* m.Ve[1:S,t+1])
                Qr[a] = m.rewards[s,a,t] + sum(m.phat[1:S, s, a, t] .* m.Vr[1:S,t+1])
            end
            bestA = indmax(Qe+Qr)
            m.policy[s, t] = bestA
            m.Ve[s, t] = Qe[bestA]
            m.Vr[s, t] = Qr[bestA]
        end
    end
end
