type ClueTask
    len::Int64
    clue_locs::Vector{Int64}
    num_colors::Int64
    H::Int64

    pos::Int64
    clues::Vector{Int64}
    t::Int64
end

function ClueTask(len, clue_locs, num_colors, H)
    ClueTask(len, clue_locs, num_colors, H, 1, rand(1:num_colors, length(clue_locs)), 1)
end

horizon(task::ClueTask) = task.H
nclues(task::ClueTask) = length(task.clue_locs)
nS(task::ClueTask) = task.len + (task.num_colors * nclues(task))
nA(task::ClueTask) = 2 + nclues(task)

function restart!(task::ClueTask, clues=nothing)
    task.pos = 1
    task.t = 1
    if clues == nothing
        task.clues = rand(1:task.num_colors, nclues(task))
    else
        task.clues = clues
    end
    observation(task, fill(false, nclues(task)))
end

function observation(task::ClueTask, visible)
    if all(visible .== false)
        res =  task.pos
    else
        res = task.len
        for i=1:nclues(task)
            if visible[i]
                res += task.clues[i]
                break
            else
                res += task.num_colors
            end
        end
    end
    res
end

function step!(task::ClueTask, a)
    visible=fill(false, nclues(task))

    r = 0. 
    for i=1:nclues(task)
        if a == 1 && task.pos == task.clue_locs[i]
            visible[i] = true
        end
        if task.pos == task.len - i
            if a == task.clues[i]
                task.pos += 1
                r = 1
            else
                r = -1
            end
        end
    end
    if task.pos < task.len - nclues(task)
        if a == nA(task)
            task.pos += 1
        elseif a == nA(task)-1 && task.pos > 1
            task.pos -= 1
        end
    end

    

    stop = task.t > horizon(task)
    observation(task, visible), r, stop
end

type ClueTaskJointObs
    task::ClueTask
    windowsize::Int64

    observations::Vector{Int64}
end

horizon(t::ClueTaskJointObs) = horizon(t.task)
nA(t::ClueTaskJointObs) = nA(t.task)
nS(t::ClueTaskJointObs) = nS(t.task)^t.windowsize
function restart!(t::ClueTaskJointObs, clues=nothing)
    fill!(t.observations, 1)
    restart!(t.task, clues)
    observe(t)
end

function observe(t::ClueTaskJointObs)
    obs = 1
    S = nS(t.task)
    for i=1:t.windowsize
        obs += (t.observations[i]-1) * S^(i-1)
    end
    obs
end
function step!(t::ClueTaskJointObs, a)
    sn, r, stop = step!(t.task, a)
    t.observations[2:end] = t.observations[1:end-1]    
    t.observations[1] = sn
    obs = observe(t)
    obs, r, stop
end

function expected_return(t::ClueTaskJointObs, detpolicy)
    args = (1:t.task.num_colors for i=1:nclues(t.task))
    res = 0.
    n = 0
    for a ∈ Base.product(args...)
        clues = collect(a)
        s = restart!(t, clues)
        ret = 0.
        for h=1:horizon(t)
            a = detpolicy(s, h)
            sn, r, stop = step!(t, a)
            observe!(m, s, a, h, r, sn)
            s = sn
            ret += r
        end
        #@show clues, ret
        res += ret
        n += 1
    end
    res / n
end

