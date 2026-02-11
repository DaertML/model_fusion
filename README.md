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

# Solutions to the problem
The point is to embed the capabilities of non-transformers into transformers; so we start with trying this with CNNs and classic ML models. Spoiler alert: different mechanisms bring us places! So this is likely a new research direction that can be taken, and you can just train smaller models, make them superhuman in a field, and then start pouring and stitching them inside other bigger models that you want to use for that certain tasks.

# Which are the benefits?
The fact that by just training a smaller specialized model you can get all you need to have:
- Special model
- Improved generalized model
without the need of training general model and then distilling, is a thing. I see a way to make better model capabilities cheaper.

# What if it doesnt work for my use case?
With this, we open a new door that can be further studied. I feel like just doing LLM finetuning, LoRAs, SFT... is moving us away from classic and good CNNs and just making us waste ton of resources to just modify attention layers... maybe you guys just forgot that the point of attention is to learn on the go from data patterns that it finds, and not just memorize text patterns, huh?

# Proposals
Gemini and I have found these mechanisms to be an interesting body of study, as ways to do the thing we wanna do (from Gemini 3, verbatim):
1. Low-Rank Adaptation (LoRA) FusionInstead of keeping the LLM entirely frozen, you use a LoRA adapter to "retrain" the LLM's attention layers specifically to understand the CNN's output.How it works: You inject small, trainable matrices ($A$ and $B$) into the LLM's self-attention blocks. These matrices are trained to transform the CNN's feature vector into "tokens" that resonate with the LLM’s existing weights.Why it's better: It prevents the "0% win rate" because the LLM slightly bends its internal logic to accommodate the new visual information, rather than just receiving a "noisy" input it doesn't understand.
2. Gated Linear Aligned Fusion (GLAF)This is a "filtering" mechanism. Instead of the LLM just seeing the CNN data, we use a Gating Layer that decides how much the CNN should influence each LLM token.The Logic:$$Output = (1 - g) \cdot \text{LLM\_hidden} + g \cdot \text{CNN\_projected}$$where $g$ (the gate) is a value between 0 and 1 learned during the training of the glue.Why it's better: In Blackjack, there are times when the LLM's "language" knowledge (logic) is more important, and times when the "visual" state (the cards) is everything. The gate learns when to listen to which model.
3. State-Space "Prompt" InjectionInstead of injecting the CNN into the middle of the LLM, you convert the CNN's output into a continuous prefix.How it works: You take the 64-dim CNN output and project it into $N$ virtual tokens (Soft Prompts). You prepend these tokens to the LLM's input sequence.The Difference: This forces the LLM to treat the card state as the "context" or "instructions" for the entire conversation.

# Experiment
I wanted to check if LLMs could play blackjack using the OpenAI gym; I tried an apparently good and recent model (qwen3:8b); and it was no good on the task at hand, so being a problem that should be easy for a model to learn, decided to take a glance at it and even make it even more difficult for the LLM to play.

So, decided to go back in time, very very back in time, and attempt this crazy journey using gpt2 (yep the base old gpt2 that is useless by today's standards); and it was able to do the task of playing blackjack with a 35/45% of Wins, by just trying the different model fusion methods that I have mentioned.

The CNN that I embed into gpt2 was trained in 10 seconds or less (on 3090), instead of the minutes/hours (havent tried but will do the test) that it would have taken to train the gpt2 model to do this.

In addition to trying CNN model trained on blackjack, I also decided to make it harder for me and attempt to do this with classic ML models; they also work, so apparently you can reuse all your classic models you have trained in the past and make your LLMs better at that task by just fusing them.

# What is the use of it?
