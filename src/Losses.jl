#
# provides various loss functions
#

module Losses

export
    loss01,
    cliplogloss,
    sqloss,
    empirical_loss

# 0-1 loss
loss01(y,z) = float(y!=z)
# define truncated log loss
clip(z) = max(min(z,0.99),0.01)
logloss(y,z) = -1.0*(y*log(10,z)+(1.0-y)*log(10,1-z))
sqloss(y,z) = (y-z)^2
# note that this is in [0,2]
cliplogloss(y,z) = logloss(y,clip(z))
clipsqloss(y,z) = min((y-z)^2,1)

function empirical_loss(labels::Array{Float64,1},predictions::Array{Float64,1},lossfct)
    return sum(map(lossfct,labels,predictions))/length(labels)
end

end
