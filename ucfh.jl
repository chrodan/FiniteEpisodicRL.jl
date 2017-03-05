type UCFH<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 4}
    phat::Array{Float64, 4}
    n::Array{UInt64, 3}
    nstemp::Array{UInt64, 4}
    ntemp::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    Vopt::Matrix{Float64}
    δ::Float64
    ϵ::Float64
    rewards_known::Bool
    update_schedule::Bool
    ln1d::Float64
    mval::Float64
    mfactor::Float64
    update_necessary::Bool
end

function UCFH(S, A, H, rewards, δ, ϵ, update_schedule=false, mfactor=1.)
    policy = zeros(UInt64, S, H)
    ns = zeros(UInt64, S, S, A, H)
    phat = zeros(S, S, A, H)
    phat[1, :, :, :] = 1.
    n = zeros(UInt64, S, A, H)
    
    δ1 = δ / (2 * S^2 * A * H * log2( S^2 * H^2 * 4 / ϵ))
    ln1d = log(6 / δ1)
    mval = 512 * (log2(log2(H)))^2 * S * H^3 * log(8 * H^2 * S^2 / ϵ)^2 * log(3 / δ1)

    m = UCFH(policy, ns, phat, n, copy(ns), copy(n), rewards, zeros(S, H+1), δ, ϵ, true, update_schedule, ln1d, mval, mfactor, false)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end
function observe!(m::UCFH, s, a, t, r, sn) 
    if !m.update_schedule         
        m.ns[sn, s, a, t] += 1
        m.n[s, a, t] +=  1
        if !m.rewards_known
            m.rewards[s, a, t] += r
        end
    else
        if m.ntemp[s,a,t] >= nS(m) * m.mval * m.mfactor * horizon(m)
            return
        end
        m.nstemp[sn, s, a, t] += 1
        m.ntemp[s, a, t] +=  1
        if m.ntemp[s,a,t] >= max(m.mval * m.mfactor, m.n[s,a,t])
            m.update_necessary = true
            m.n[s,a,t] = m.ntemp[s,a,t]
            m.ns[:,s,a,t] = m.nstemp[:, s, a, t]
        end
    end
end

"""computes the convex hull of the confidence set stated in the paper
the convex hull is sufficient for the theoretical results to hold"""
function confidence_set(m::UCFH, p, n)

    ln1d = m.ln1d
    if n <= 1
        return (0., 1.)
    end
    empstd = √(p * (1 - p))
    hoeff = √(ln1d / 2 / n)
    bern = empstd * √(ln1d * 2 / n) + 7 / 3 / (n - 1) * ln1d
    
    pmax = min(1, p + hoeff, p + bern)
    pmin = max(0, p - hoeff, p - bern)

    stdC =√(2 * ln1d / (n - 1))
    # confidence interval induced by empricial stddev bound
    if empstd >= stdC && empstd <= 0.5
        v =  √(0.25 - (empstd - stdC)^2)
        pmin = max(pmin, .5 - v)
        pmax = min(pmax, .5 + v)
    end

    ## potentially nonconvex confidence set induced by empiricla stddev bound
    #if empstd <= 0.5 - stdC
    #    p1 = 0.5 - √(0.25 - (empstd + stdC)^2)
    #    p2 = 0.5 + √(0.25 - (empstd + stdC)^2)
    #    if p1 <= pmin
    #        pmin = max(pmin, p2)
    #    elseif p2 >= pmax
    #        pmax = min(pmax, p1)
    #    end
    #end
    (pmin, pmax)    
end

function update_policy!(m::UCFH)
    if m.update_schedule && !m.update_necessary
        return
    end
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    Q = zeros(A)
    m.rewards_known || error("UCFH assumes known rewards")

    pcur = zeros(S)
    pmax = ones(S)
    
    m.phat .= m.ns ./ reshape(m.n, 1, S, A, H) 
    for t=H:-1:1
        V = m.Vopt[:, t+1]
        ind = sortperm(V)

        for s ∈ 1:S
            for a ∈ 1:A
                mass_left = 1.
                EV = 0.
                for i ∈ 1:S
                    pcur[i], pmax[i] = confidence_set(m, m.phat[i, s, a, t], m.n[s,a,t])
                    mass_left -= pcur[i]
                    EV += pcur[i] * V[i]
                end
                if mass_left < 0
                    error("Inadmissible interval with total mass $(1 - mass_left)")
                end
                for i ∈ S:-1:1
                    ii = ind[i]
                    pcmax = min(mass_left+pcur[ii], pmax[ii])
                    mass_left -= pcmax - pcur[ii]
                    EV += (pcmax - pcur[ii]) * V[ii]
                    pcur[ii] = pcmax
                    if mass_left < 3 * eps(typeof(mass_left))
                        break
                    end
                end
                                
                r = m.rewards[s, a, t]
                
                Q[a] = r + EV
            end
            bestA = indmax(Q)
            m.policy[s, t] = bestA
            m.Vopt[s, t] = Q[bestA]
        end
    end
    m.update_necessary = false
end
