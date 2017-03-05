type MedianPAC<:FiniteHorizonAgent
    policy::Matrix{UInt64}
    ns::Array{UInt64, 5}
    nsactive::Array{UInt64, 5}
    nactive::Array{UInt64, 3}
    nsnew::Array{UInt64, 4}
    cbucket::Array{UInt64, 3}
    #n::Array{UInt64, 3}
    rewards::Array{Float64, 3}
    #Vopt::Matrix{Float64}
    #δ::Float64
    ϵa::Float64
    maxobs::Float64
    maxRet::Float64
    maxR::Float64
    keep_updating::Bool
    trash_samples::Bool
end

function MedianPAC(S, A, H, rewards, δ, ϵ, keep_updating=false, trash_samples=false)
    policy = zeros(UInt64, S, H)
    maxRet = sum(maximum(reshape(rewards, S*A, H), 1))
    # set to satisfy ϵ0 = ϵa / H in Theorem 5.4
    ϵa = 1/2*H*ϵ
    km = ceil(Int64, 50 / 9 * log(4 / δ * 2 * log2(2 * maxRet / ϵa * S * A * H)))
    nbuckets = km
    ns = zeros(UInt64, S, nbuckets, S, A, H)
    nsnew = zeros(UInt64, nbuckets, S, A, H)
    nsactive = zeros(UInt64, S, nbuckets, S, A, H)
    nactive = zeros(UInt64, S, A, H)
    cbucket = ones(UInt64, S, A, H)
   
    # k in the paper
    maxobs = km * 2^ceil(Int64,log2(4 * maxRet^2 / ϵa^2))
    
    m = MedianPAC(policy, ns, nsactive, nactive, nsnew, cbucket, rewards, ϵa, maxobs, 
                  maxRet, maximum(rewards[:]), keep_updating, trash_samples)
    rand!(m.policy, 1:A) # Initialize with uniformly random policy
    m
end

nbuck(m::MedianPAC) = size(m.ns, 2)
function observe!(m::MedianPAC, s, a, t, r, sn) 
    if !m.keep_updating && m.nactive[s,a,t] >= m.maxobs
        return
    end
    m.nsnew[m.cbucket[s, a, t], s, a, t] = sn
    m.cbucket[s,a ,t] += 1
    if m.cbucket[s, a, t] > nbuck(m)
        m.cbucket[s,a,t] = 1
        for i ∈ 1:nbuck(m)
            m.ns[m.nsnew[i, s, a, t], i, s, a, t] += 1
        end
        if sum(m.ns[:,1,s,a,t]) >= max(1, 2 * m.nactive[s, a, t])
            # new batch of twice the size, replace existing batch
            m.nsactive[:,:, s,a ,t] = m.ns[:,:,s, a, t]
            m.nactive[s, a, t] = sum(m.ns[:,1, s, a, t])
            if m.trash_samples
                m.ns[:,:,s,a,t] = 0
            end
        end
    end
end
maxV(m::MedianPAC) = m.maxRet
maxR(m::MedianPAC) = m.maxR

function update_policy!(m::MedianPAC)
    S = nS(m)
    H = horizon(m)
    A = nA(m)
    ϵb = m.ϵa * sqrt(m.maxobs)
    Vmax = maxV(m)
    Rmax = maxR(m)
    
    Q = zeros(A)
    V = zeros(S)
    Vnew = zeros(S)
    means = zeros(nbuck(m))
    for t=H:-1:1
        V, Vnew = Vnew, V
        curmaxV = min(maximum(V) + Rmax, Vmax)
        for s ∈ 1:S
            for a ∈ 1:A
                if m.nactive[s,a, t] == 0
                    Q[a] = curmaxV
                    break
                end
                slack = ϵb / √(m.nactive[s, a, t])
                for i ∈ 1:nbuck(m)
                    means[i] = V ⋅ m.nsactive[:,i, s, a, t] / m.nactive[s, a, t]
                end
                Q[a] = min(curmaxV, slack + m.rewards[s, a, t] + median(means))
            end
            bestA = indmax(Q)
            m.policy[s, t] = bestA
            Vnew[s] = Q[bestA]
        end
    end
end
