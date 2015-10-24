---
layout: default
title: Investigating Relationships Between Parts in CNNs
published: false
---

Investigating Relationships Between Parts in CNNs
===






outline
---
{::comment}
TODO 
{:/comment}
* introduction
    * background, which leads to...
    * goal: investigating relations between
* review visualization methods


$$
Ax = b
$$



Introduction
---

conv nets are great... successes... pointers to more info
...but there's a problem.

We don't understand how they work. Obviously, we understand them completely in some
sense (we created them), but we don't understand them very well in another sense.
You can see an example of something we understand to a similar degree in [Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)),
IBM's computer which famously beat the greatest chess player of the time.
At some point a team of programmers coded the thing up. They clearly know

use the example of deep blue and the guy who sat at the chess board and output deep blue's moves


.... but it's harder in this case because people can write a book about how to play chess,
then you might come to understand it after reading and practicing. Nobody can write a book
about how to see. It would wrong to say we don't understand the thing at all.
In fact, you can probably start to describe how you do recognize a cat. Give it a try.
Look at the picture of a cat below and try to figure out why you think it's a cat.

TODO: put picture of cat

Done? Excellent! Here's what I said:

> This picture has some pointy ears. They're triangle shaped on the top and the
> outline has a furry look to it. Below the ears are a pair of eyes.
> TODO: finish

Your description probably 

Lot's of ideas have been

It's just something everyone with working eyes does. 



... the cool thing about CNNs is that __there is complexity__, we may be able
to understand that complexity if we enlist the aid of our visual senses,
and it might even be close to how we understand the visual world (it discovers parts and we tried to describe the world in parts).
This is not at all the case for Deep Blue.
That program worked by approximately enumerating possible chess games
many moves away (there are lots, so this is hard) and applying some
algorithmic and heursitic tricks so it could enumerate fewer.
When humans play chess we mainly look for patterns and can make only
extremely limited progress trying to consider all possible future moves.


---


Deep Convolutional Networks (DCNs) have had a major impact on Computer Vision
and Machine Learning in the past couple years[^1].



There have been many approaches to visualizing CNNs, but they all focus
on the same question:

> What do a particular set of neurons mean in image space?

Deep neural networks are supposed to learn high level information,
so this is a good question to ask.
To some degree, visualizations have been able to answer this question.
They can sometimes confirm that particular neurons or sets of neurons
consistently respond to concepts like "dog's" head and not so much to
other concepts[^2]. Here's another (ill posed) question we might be
interested in answering:

> Why is this a cat?

Activation based visualizations will say this is a cat because the
neuron whose visualization looks like a pointy ear fired. And the
neuron which sees furry things fired and so did the neuron that
detects paws. But why did those neurons fire?
Clearly we know that certain



Reasons to study connection visualizations:

* The question "Why is this a cat?" is just asking why a particular
  neuron's activation is high. We can also ask why other neurons
  have high activation ("Why is this a cat's eye?"). We can't answer
  similar the questions for these other neurons in the same manner
  as we answered the first question because there aren't any
  labeled lower level neurons.

* Perhaps the cat (TODO: better example; bus/car?) is only a cat because of context.
  Again, this is great to know, but it breaks down between gabor filters (conv1)
  and cat faces (conv4/conv5).
  Could connection visualizations help reveal this?

* Neuron visualizations don't give a sense of compositionality. How is layer l
  activation computed from layer l-1 activation.

* We don't really understand middle layers of neurons because we have nothing
  to relate them to. We can relate them to the higher and lower layers
  of visualizations, so perhaps by doing so we'll be able to understand them.





Points to make / Questions to ask
---

* Why do we need to modify the backward step to get better visualizations?

* There might be other ways than the weight matrix method to relate
  neurons to eachother, also ways to relate sets of neurons.

* There are probably much better ways to visualize this information,
  but I just wanted to do a simple visualization here.

* Maybe we can begin to answer questions like
  "What makes this dog detector different from this other dog detector?"

* It's clear that the hierarchy of parts that makes up the eye detector
  decreases in "semantic complexity" from layer to layer; one layer
  is clearly simpler than the one above it.

* Is my cropping method similar to the "empirical recptive fields" of the
  emerging detectors paper?

* Note that I'm not (yet) talking about relating spatial location.

* Note that the software I wrote doesn't just have to consider weights
  between adjacent layers, it can weight any pair of layers (just using
  gradients from a fixed starting point).

* I need to make a point about how good examples drive new ideas. (cite Bostock's talk?...)

* Interface between conv5 and fc6




[^1]: See [this](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/) nice introduction by Chris Olah 

[^2]: emerging object detectors (TODO)
