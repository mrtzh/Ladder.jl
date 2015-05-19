module Ladder
using Losses

export Submission
export LadderState
export ladder
export score!

type Submission
    id::String
    team::String
    labels::Array{Float64,1}
end

Submission(a::Array{Float64,1}) = Submission("","",a)

abstract LadderState

type FreeLadder <: LadderState
    labels::Array{Float64,1}
    current::Float64
    leader::Submission
end

type FixedLadder <: LadderState
    labels::Array{Float64,1}
    step_size::Float64
    current::Float64
    leader::Submission
end

function ladder(labels)
    return FreeLadder(labels,Inf,Submission(Float64[]))
end

function ladder(labels,step_size)
    return FixedLadder(labels,step_size,Inf,Submission(Float64[]))
end

# evalulate submission on parameter free ladder instance
function score!(l::FreeLadder,sub::Submission,lossfct)
    loss = empirical_loss(l.labels,sub.labels,lossfct)
    s = 0.0
    if length(l.leader.labels) == 0
        # no submission yet
        s = std(map(lossfct,l.labels,sub.labels))
    else
        # compute paired standard deviation between previous best and given labels
        s = std(map(lossfct,l.labels,sub.labels)-map(lossfct,l.labels,l.leader.labels))
    end
    # one sided paired t-test
    if loss < l.current - s/sqrt(length(l.labels))
        l.current = loss
        l.leader = sub
    end
    return l.current
end

# evaluate submision on ladder instance with fixed step size
function score!(l::FixedLadder,sub::Submission,lossfct)
    loss = empirical_loss(l.labels,sub.labels,lossfct)
    if loss < l.current - l.step_size
        l.current = round(loss,int(log(10,length(l.labels))))
        l.leader = sub
    end
    return l.current
end

end # module
