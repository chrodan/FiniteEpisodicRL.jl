type MoRMax<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    
    ns::Array{UInt64, 4}
    n::Array{UInt64, 3}
    
    phat::Array{Float64, 4}
    rhat::Array{Float64, 3}
    
    pprime::Array{Float64, 4}
    rprime::Array{Float64, 3}
    
    rewards::Array{Float64, 3}

    τ::Array{Int64, 3}
    τstar::Int64

    Qhat::Array{Float64, 3}
    Qprime::Array{Float64, 3}
    policyprime::Matrix{UInt64}
    Qprev::Array{Float64, 3}

    δ::Float64
    ϵ::Float64
    rewards_known::Bool
    maxRet::Float64
    mfactor::Float64
end

function MoRMax(S, A, H, rewards, δ, ϵ, mfactor=1.)
    policy = zeros(UInt64, S, H)
    policyprime = copy(policy)

    ns = zeros(UInt64, S, S, A, H)
    n = zeros(UInt64, S, A, H)
    
    phat = zeros(S, S, A, H)
    pprime = copy(phat)

    maxrs = maximum(reshape(rewards, S*A, H), 1)
    rhat = copy(rewards)
    for t=1:H
        rhat[:, :, t] = sum(maxrs[t:end])
    end
    rprime = copy(rhat)
    τ = zeros(Int64, S, A, H)

    Qhat = zeros(S, A, H)
    Qprime = copy(Qhat)
    Qprev = zeros(S, A, H)
    compute_optQ!(Qhat, policy, phat, rhat)
    rand!(policy, 1:A) # Initialize with uniformly random policy
    
    MoRMax(policy, ns,n,  phat, rhat, pprime, rprime, rewards, 
           τ, 1, Qhat, Qprime, policyprime, Qprev, δ, ϵ, true, sum(maxrs), mfactor)
end


maxV(m::MoRMax) = m.maxRet
function mval(m::MoRMax)
    S = nS(m)
    A = nA(m)
    H = horizon(m)
    mval = 1800*maxV(m)^2 / m.ϵ^2 * H^2
    mval *= log(144 * H * S^2 * A^2 * maxV(m) / m.δ / m.ϵ)  
    mval
end


function update_policy!(m::MoRMax)
    S = nS(m)
    A = nA(m)
    H = horizon(m)
    mvalue = mval(m)
    if !m.rewards_known
        error("Not Implemented")
    end

    for t ∈ 1:H, a ∈ 1:A, s ∈ 1:S
        if m.n[s,a,t] >= mvalue * m.mfactor
            
            if (m.τ[s, a, t] == 0 || 
                (m.τstar > m.τ[s,a,t] && m.Qprev[s, a, t] > m.ϵ / H + m.Qhat[s, a, t]))
                    copy!(m.pprime, m.phat)
                    copy!(m.rprime, m.rhat)
                    m.pprime[:, s, a, t] .= m.ns[:, s, a, t] / m.n[s, a, t]
                    m.rprime[s, a, t] = m.rewards[s, a, t]

                    #println("Propose model update for s=$s, a=$a, t=$t")
                    compute_optQ!(m.Qprime, m.policyprime, m.pprime, m.rprime)
                    if m.Qprime[s, a, t] <= m.Qhat[s, a, t]
                        m.phat[:, s, a, t] = m.pprime[:, s, a, t]
                        m.rhat[s, a, t] = m.rprime[s, a, t]
                        copy!(m.Qhat, m.Qprime)
                        copy!(m.policy, m.policyprime)
                        #println("Accepted model update for s=$s, a=$a, t=$t")
                    end
                    m.τstar += 1
                    m.τ[s,a,t] = m.τstar
                    m.Qprev[s,a,t] = m.Qhat[s,a,t]
            end

            m.n[s,a,t] = 0
            m.ns[:, s, a, t] = 0
        end
    end
end

function compute_optQ!(Q, pol, P, R)
    S, A, H = size(R)
    V = zeros(S)
    for t ∈ H:-1:1
        Q[:, :, t] .= R[:, :, t] + squeeze(sum(V .* P[:, :, :, t], 1), 1)
        for s ∈ 1:S
            a = indmax(Q[s, :, t])
            pol[s, t] = a
            V[s] = Q[s, a, t]
        end
    end
end
