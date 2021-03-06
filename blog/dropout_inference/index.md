---
layout: blog
title: Approximate Inference for Deep Networks with Dropout
comments: true
published: false
use_math: true
---

Approximate Inference  for Deep Networks with Dropout
===

One of the first things you're likely to learn when you get in to deep learning
is Dropout. Deep networks can grow quite large because now we have GPUs that
can run them quickly, there's enough data to train them, and bigger (especially
deeper) networks empirically tend to work better. Of course, networks with so
many parameters are part of a much larger model space than other models, so they
tend to overfit. Among the most common and effective ways to stop this is the
application of dropout.

Instead of directly training a given network, Dropout augments the network
during each iteration of training by randomly removing hidden activations.
If you think of hidden activations as feature detectors for the rest of the
network (which sits on top) then this can be thought of as encouraging
a network to not rely too strongly on one feature. It also discourages
neurons from being highly co-adapted, and finally it can be interpreted as training
a different model at each iteration.

TODO: show dropout picture

I'd like to focus on that last interpretation. In particular, if we're training
different networks at different iterations then how do we average them? As
it turns out, the proposed trick of muliplying weights by $$\frac{1}{p}$$
works out nicely. It provides an approximation to averaging _all_ of the
exponentially many models in the same time it takes to run inference for
a single model! I've seen some justification of this before, but only
for the case of logistic regressors and I can't find the original source.
If you know of the source, please let me know and I'll be sure to make a note here.

First, consider dropping out inputs of a logistic regressor.
Here, the different models are logistic regressors with all possible subsets
of inputs turned off, so there are $$2^N$$.
In this case, we can actually exactly average all those models in the time
it take to run one! To see this, first think of Dropout as matrix multiplication.
Let $$Z = ...$$,

$$
    \sigma(x) = W^T X   \text{ vs }  \sigma(x) = W^T Z X
$$

I'm interested in

$$
    E_Z[\sigma(x)] = ... = W^T E_Z[Z] X = W^T \frac{1}{p} X
$$

To do exact inference in this model I only need to multiply $$W$$ by $$\frac{1}{p}$$!
It gets a bit better. I can expand this to deep models.

....briefly mention that exact inference can be done when dropping out any single ReLU layer of a deep model...

... talk about the layer-wise view of actual dropout inference...


Vs Gal/Gharamani: They extract uncertainty estimates from Dropout (estimate the mean output and its variance using a monte-carlo procedure)
