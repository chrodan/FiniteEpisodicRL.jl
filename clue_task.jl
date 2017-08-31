type ClueTask
    len::Int64
    clue_locs::Vector{Int64}
    num_colors::Int64
    H::Int64

    pos::Int64
    clues::Vector{Int64}
    t::Int64
end

horizon(task::ClueTask) = task.H
nclues(task::ClueTask) = length(task.clue_locs)
nS(task::ClueTask) = task.length * ((task.num_colors+1)^nclues(task))
nA(task::ClueTask) = 2 + nclues(task)

function restart!(task::ClueTask)
    task.pos = 1
    task.t = 1
    task.clues = rand(1:task.num_colors, nclues(task))
    observation(task, fill(false, nclues(task)))
end

function observation(task::ClueTask, visible)
    res = task.pos - 1
    expon = 0
    for i=1:nclues(task)
        expon += (task.num_colors+1)^(i-1)*visible[i]*task.clues[i]
    end
    @show expon
    res += task.len*expon + 1 
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
                r += 1
            end
        end
    end
    if task.pos < task.len - nclues(task)
        if a == nA(task)
            task.pos += 1
        elseif a == nA(task)-1 && task.pos > 0
            task.pos -= 1
        end
    end

    

    stop = task.t > horizon(task)
    observation(task, visible), r, stop
end
