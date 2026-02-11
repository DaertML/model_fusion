# model_fusion
What if we could train tiny models and fuse them with any other model?

# Introduction
I've been training models for a while now; it is not the most repeated task in my day to day, but there is a big fraction of GPU cycles that I spend training.
One of the problems that I have recently encountered is the fact that small models in niche problems can be trained very fast and easily, without the need of spending a lot of money and resources.
That triggered a need: what if I could train a lot of many small models and fuse them into bigger ones, so that, we could take the best of both worlds: general models that use specialized models.
The second need got triggered a while ago: LLMs alone will not bring us to AGI; there are many problems that need to be solved in a real time fashion, and solving them with LLMs may become real time with better hardware, but wont hide the fact that they will run in a costly manner, and will take a considerable amount of resources to train.
Thus, I got a bitter feeling on being unable to use my favourite AI interface: LLM models, with many of the models that I have trained in the past, and that work extremely well, and that do not make use of transformers.

# Problem to solve
There are a lot of models that are not LLMs, that have become experts on their own field: we have self driving models, models that play games like DOTA, Atari, Chess, Go... and yet, we neglect their capabilities because we need to be on the wave of training LLMs, and having better LLMs is the only thing we care about.
I felt that it could be amazing if I could take a non transformer model, and a non LLM model and use it inside an LLM transformer, so that, the transformer got to learn the best behavior from it, yet without retraining the LLM.
It feels like if we can find a way of merging (fusing!) models that have been in production and that are very efficient to improve the capabilities of LLMs, we may get away with retraining LLMs. Training LLMs is expensive, training tiny models like CNNs is not (cheaper at least!), thus, training small models and stitching them inside the LLMs may give us LLMs with better capbilities without training LLMs.
The "without training" is a bit of a lie, as one of the proposed methods is "LoRA"; but hey, we are stitching models! It is not even distillation when the model to distill from is smaller right? (that's another study we need to do at some point).

# 
