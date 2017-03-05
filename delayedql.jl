
type DelayedQL<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    
    rewards::Array{Float64, 3}

    b::Array{UInt64, 3}
    l::Array{Int64, 3}
    learn::Array{Bool, 3}
    tstar::UInt64
    tval::UInt64

    Q::Array{Float64, 3}
    AU::Array{Float64, 3}

    δ::Float64
    ϵ::Float64
    maxRet::Float64
    mfactor::Float64
    immediate_updates::Bool
end


function DelayedQL(S, A, H, rewards, δ, ϵ, mprecond=1.)
    policy = zeros(UInt64, S, H+1)
    b = zeros(UInt64, S, A, H)
    l = zeros(UInt64, S, A, H)
    AU = zeros(Float64, S, A, H)
    learn = zeros(Bool, S, A, H)
    maxrs = maximum(reshape(rewards, S*A, H), 1)
    V_max = sum(maxrs)
    Q = zeros(Float64, S, A, H+1)
    for t=1:H
        Q[:,:,t] = sum(maxrs[t:H])
    end
    eps1 = ϵ / 3 / H 
    mfactor = mprecond * ( 1 + V_max)^2 / 2 / eps1^2 * log(3 * S * A * H / δ *(1 + S * A * H / eps1 ))
    rand!(policy, 1:A) # Initialize with uniformly random policy
    DelayedQL(policy, rewards, b, l, learn, 0, 0, Q, AU, δ, ϵ, V_max, mfactor, true)
end

function observe!(m::DelayedQL, s, a, t, r, sn) 
    m.tval += 1
    if m.b[s,a,t] <= m.tstar
        m.learn[s,a,t] = true
    end
    if m.learn[s,a,t]
        if m.l[s,a,t] == 0
            m.b[s,a,t] = m.tval
        end
        m.l[s,a,t] += 1
        m.AU[s,a,t] += m.Q[sn,m.policy[sn,t+1], t+1] + m.rewards[s,a,t]
        if m.immediate_updates 
            attempt_update!(m, s,a,t)
        end
    end
end

function attempt_update!(m::DelayedQL, s, a, t)
    if m.l[s,a,t] >= m.mfactor
        if m.Q[s,a,t] >= m.AU[s,a,t] / m.l[s,a,t] + 2 * ϵ1(m)
            m.Q[s,a,t] = m.AU[s,a,t] / m.l[s,a,t] + ϵ1(m)
            m.policy[s,t] = indmax(m.Q[s, :, t])
            m.tstar = m.tval
        elseif m.b[s,a,t] > m.tstar
            m.learn[s,a,t] = false
        end
        m.l[s,a,t] = 0
        m.AU[s,a,t] = 0
    end
end

ϵ1(m::DelayedQL) = m.ϵ / 3 / horizon(m)
update_policy!(m::DelayedQL) = nothing

