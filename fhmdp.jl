using Distributions
type SmallFiniteHorizonMDP
    P::Array{Float64, 4}
    atbls::Array{Distributions.AliasTable,3}
    R::Array{Float64, 3}
    p0::Categorical
    s::UInt64
    t::UInt64
end
function SmallFiniteHorizonMDP(P, R, p0, s, t)
    S, A, H = size(R)
    atbls = Array{Distributions.AliasTable, 3}(S, A, H)
    for t=1:H, a=1:A, s=1:S
        atbls[s,a,t] = Distributions.AliasTable(P[:, s, a, t])
    end
    SmallFiniteHorizonMDP(P, atbls, R, p0, s, t)
end


horizon(mdp::SmallFiniteHorizonMDP) = size(mdp.R, 3)
nS(mdp::SmallFiniteHorizonMDP) = size(mdp.R, 1)
nA(mdp::SmallFiniteHorizonMDP) = size(mdp.R, 2)
rewards(mdp::SmallFiniteHorizonMDP) = mdp.R
function restart!(mdp::SmallFiniteHorizonMDP)
    s = rand(mdp.p0)
    mdp.s = s
    mdp.t = 1
    s
end
function step!(mdp::SmallFiniteHorizonMDP, a)
    r = mdp.R[mdp.s, a, mdp.t]
    sn = rand(mdp.atbls[mdp.s, a, mdp.t])
    mdp.s = sn
    mdp.t += 1
    stop = mdp.t > horizon(mdp)
    sn, r, stop
end

function expected_return(mdp::SmallFiniteHorizonMDP, policy)
    H = horizon(mdp)
    S = nS(mdp)
    A = nA(mdp)
    
    Vold = zeros(S)
    Vnew = zeros(S)
    
    for t=H:-1:1
        for s=1:S
            a = policy(s, t)
            Vnew[s] = mdp.R[s, a, t] + mdp.P[:, s, a, t] ⋅ Vold
        end
        Vnew, Vold = Vold, Vnew
    end
    mdp.p0.p ⋅ Vold
end

function optimal_policy(mdp::SmallFiniteHorizonMDP)
    H = horizon(mdp)
    S = nS(mdp)
    A = nA(mdp)
    policy = zeros(UInt64, S, H)
    Vold = zeros(S)
    Qnew = zeros(S, A)
    
    for t=H:-1:1
        Qnew = mdp.R[:, :, t] + squeeze(sum(Vold .* mdp.P[:, :, :, t], 1), 1)
        for s=1:S
            policy[s, t] = indmax(Qnew[s, :])
            Vold[s] = Qnew[s, policy[s, t]]
        end
    end
    (s, t) -> policy[s, t]
end

function value_fun(mdp::SmallFiniteHorizonMDP, policy)
    H = horizon(mdp)
    S = nS(mdp)
    A = nA(mdp)
    
    V = zeros(S, H+1)
    
    for t=H:-1:1
        for s=1:S
            a = policy(s, t)
            V[s, t] = mdp.R[s, a, t] + mdp.P[:, s, a, t] ⋅ V[:, t+1]
        end
    end
    V
end

gammarand(a) = if a >= 1
            rand(Distributions.GammaGDSampler(Gamma(a)))
        else
            rand(Distributions.GammaGSSampler(Gamma(a)))
        end

function dirrand!(res, αs)
    ss = 0.
    for (i, α) in enumerate(αs)
        b = gammarand(α)
        ss += b
        res[i] = b
    end
    res ./= ss
end
function dirrand(αs)
    res = zeros(length(αs))
    dirrand!(res, αs)
    res
end
function randomMDP(S, A, H, α =.1)
    dir = Dirichlet(S, α)
    P = zeros(S, S, A, H)
    R = rand(S, A, H)
    R[rand(S, A, H) .<= 0.85] = 0.
    p0 = Categorical(dirrand(repeated(α, S)))
    for s=1:S, a=1:A, t=1:H
        P[:, s, a, t] = dirrand(repeated(α, S))
    end
    SmallFiniteHorizonMDP(P, R, p0, 1, 1)
end
function stateDepTestMDP(N, ϵ=1e-2)
    S = 2N + 2
    A = 2
    H = 2
    pp0 = zeros(S)
    pp0[1] = 1
    p0 = Categorical(pp0)
    R = zeros(S, A, H)
    R[(S - N + 1):end, :, 2] = 1
    R[2, :, 2] = 0.5 + ϵ
    P = zeros(S, S, A, H)
    for s=1:S, a=1:A
        P[s, s, a, 2] = 1.
    end
    P[3:end, 1, 1, 1] = 1 / 2 / N
    P[2, 1, 2, 1] = 1.
    SmallFiniteHorizonMDP(P, R, p0, 1, 1)
end


type ChainMDP
    P::Array{Float64, 3}
    R::Vector{Float64}
    N::Int64
    s::UInt64
    t::UInt64
    r0::Float64
end
nS(m::ChainMDP) = m.N
nA(m::ChainMDP) = 2
horizon(m::ChainMDP) = m.N
function rewards(m::ChainMDP)
    R = zeros(nS(m), nA(m), horizon(m))
    R .= m.R
    R
end

function ChainMDP(N, r0=0.) 
    P = zeros(N, N, 2)
    for s=1:N
        sn = max(s - 1, 1)
        P[sn, s, 1] = 1.
        P[sn, s, 2] = 1./N
        sn = min(s+1, N)
        P[sn, s, 2] = 1 - 1./N
    end

    R = zeros(N)
    R[end] = 1.
    R[1] = r0
    ChainMDP(P, R, N, 1, 1, r0)
end
function restart!(mdp::ChainMDP)
    mdp.s = 1
    mdp.t = 1
    1
end
function step!(mdp::ChainMDP, a)
    r = 0.
    if mdp.s == mdp.N
        r = 1.
    elseif mdp.s == 1
        r = mdp.r0
    end
    sn = max(mdp.s-1, 1)
    if a == 2 && rand() > 1/mdp.N
        # go right
        sn = min(mdp.s+1,mdp.N)
    end
    mdp.s = sn
    mdp.t += 1
    stop = mdp.t > horizon(mdp)
    sn, r, stop
end

function expected_return(mdp::ChainMDP, policy)
    H = horizon(mdp)
    S = nS(mdp)
    A = nA(mdp)
    
    Vold = zeros(S)
    Vnew = zeros(S)
    
    for t=H:-1:1
        for s=1:S
            a = policy(s, t)
            Vnew[s] = mdp.R[s] + mdp.P[:, s, a] ⋅ Vold
        end
        Vnew, Vold = Vold, Vnew
    end
    Vold[1]
end

function optimal_policy(mdp::ChainMDP)
    H = horizon(mdp)
    S = nS(mdp)
    A = nA(mdp)
    policy = zeros(UInt64, S, H)
    Vold = zeros(S)
    Qnew = zeros(S, A)
     
    for t=H:-1:1
        Qnew = reshape(mdp.R, S, 1) .+ squeeze(sum(Vold .* mdp.P, 1), 1)
        for s=1:S
            policy[s, t] = indmax(Qnew[s, :])
            Vold[s] = Qnew[s, policy[s, t]]
        end
    end
    (s, t) -> policy[s, t]
end
