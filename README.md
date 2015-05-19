# Ladder.jl

A realiable leaderboard for machine learning competitions

MIT Licensed. See `LICENSE.md`.

## Installation

Open a Julia prompt and call: `Pkg.clone("https://github.com/mrtzh/Ladder.jl.git")`

## Background

See this [blog post](http://blog.mrtz.org/2015/03/09/competition.html) for a discussion on the problem of overfitting to the public leaderboard in a data science competition.

This is the code repository for [this paper](http://arxiv.org/abs/1502.04585). Here's a bibtex reference:
```
@article{BH15,
  author    = {Avrim Blum and Moritz Hardt},
  title     = {The Ladder: {A} Reliable Leaderboard for Machine Learning Competitions},
  journal   = {CoRR},
  volume    = {abs/1502.04585},
  year      = {2015},
  url       = {http://arxiv.org/abs/1502.04585},
  timestamp = {Mon, 02 Mar 2015 14:17:34 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/BlumH15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
If you use the code, we encourage you to cite our paper.

## Examples
```
The basic usage is as follows:
```
using Ladder
# these are the labels corresponding to your holdout data set
holdoutlabels = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
# create ladder instance around holdout labels
l = ladder(holdoutlabels)
# create submission
submission1 = Submission("sub1","teamA",[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
score!(l,submission1,Ladder.loss01) # returns: 0.6666666666666666
# create another submission
submission2 = Submission("sub2","teamA",[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
score!(l,submission2,Ladder.loss01) # returns: 0.6666666666666666
# Ladder judged that there was no significant improvement
# create another submission
submission3 = Submission("sub3","teamA",[1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
score!(l,submission2,Ladder.loss01) # 0.3333333333333333
```
See `examples/photo.jl` for a comprehensive example on Kaggle's Photo Quality Prediction challenge. The data set is not yet available, but will most likely be released by Kaggle in the near future.

## Other usage

You can also use the Ladder mechanism to keep track of your own progress in a data science project and avoid overfitting to your holdout set. This can be useful in situations where you repeatedly evaluate candidate models against a holdout set.
