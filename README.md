# An AI agent learning to walk in gym's BipedalWalker environment.

This AI agent uses Evolution Strategies and deep learning models to learn how to walk.

Read [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/) from OpenAI if you are interested.

After a few hundred iterations, he can walk! Or he may run ;)

![demo](https://image.ibb.co/c2c9F5/ezgif_com_resize.gif)


# Dependencies

- [evostra](https://github.com/alirezamika/evostra)
- [gym](https://github.com/openai/gym)


# Usage

To see the agent walking:

```
from bipedal import *

agent = Agent()

# the pre-trained weights are saved into 'weights.pkl' which you can use.
agent.load('weights.pkl')

# play one episode
agent.play(1)
```

To start training the agent:

```
# train for 100 iterations
agent.train(100)
```
