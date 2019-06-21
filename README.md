[//]: # (Image References)

[image1]: walking.gif "Trained Agent"
[image2]: couverture.png "Tetrapode"



# Tetrapode-robot-learning-to-walk-thanks-to-Deep-Reinforcement-Learning
In this project, I try to teach the robot to walk in a Unity simulation using the PPO (Proximal Policy Optimization) algorithm for Deep Reinforcement Learning.

![trained Agent][image1]

### Introduction
This project follows a school project where we had to design and build a four-legged robot capable of walking in a straight line. This robot consists of a 30 cm long body and four 30 cm arms, each containing three servomotors. To do this, we designed a model of each required part using Catia software, which we then printed in 3D. Once these parts were assembled with the 12 dynamixel motors, we programmed them using an OpenCM card that is similar to an Arduino card. The programming of the walk was done by following two different solutions. The first consists of a polynomial interpolation of the motor angles according to the position of the tool marks (at the end of the arm). The second is to create a simulation of the robot using Matlab's Robotics Toolbox module.

Translated with www.DeepL.com/Translator

![Tetrapode Robot][image2] 

### Deep Reinforcement learning
Now, in this project, I tried a third solution which is called [Deep Reinforcement Learning](https://en.wikipedia.org/wiki/Deep_reinforcement_learning). Basically it give the ability to the agent to distinct what combination of action seems to lead to good results from which ones leads to bad ones.
To begin, I have modified the [crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment  to create my own environment where I could train the tetrapode.
Then, I tried to train the robot using the DDPG algorithm as I used in the [continuous control project](https://github.com/GeraudMM/Continuous-Control-with-DeepRL/blob/master/README.md) from Udacity but it wasn't working very well so I tried the [PPO algorithm](https://github.com/ZeratuuLL/Reinforcement-Learning/blob/master/Continuous%20Control/Crawler/Crawler.ipynb) from [Lifeng Wei](https://github.com/ZeratuuLL) and it directly led to better results. Then I modified the environment and the algorithm slightly by slightly in order to get better results. 

### Issues and possible solutions
The problem here, though, is that the physic of a tetrapod robot is quit hard and I found that the simulation of Unity couldn't led me to expoitable results for implementation in real life. 

To solve this problem, I can think to three different solutions:
- At first, we could just try to change the software to a more complex one when talking about physics as OpenAI used for their spider
-or, we could try to give robustesse(non anglais) to the robot by training its neural networks on various slightly differentes physic reality like [OpenAI](https://openai.com/) did with [this hand](https://openai.com/blog/learning-dexterity/).
-Finally, we could directly train the neural network in real life like [here](https://www.youtube.com/watch?v=V05SuCSRAtg). Though it must require the minimum of human intervention so that we can let it train almost alone.

Finally, the simulation is interesteing because even if it's not physicly perfect, it can show us various way to program an efficient walk for the robot.
Then don't forget that in simulation the robot has many captors:
-
-
-
-
-
-

so if we were to implement that IRL we would have to add sensors on the robots.


### Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Files
The `model.py` and `ddpg_agent.py` files are from the folder on ddpg-pendulum made by Udacity.

The `DDPG_continuous_control.ipynb` notebook is the part where we can train and watch a "smart" agent evolving in the Unity environment.

The `checkpoint_actor.pth` and `checkpoint_critic.pth` files are saving of the weights of the locals actor and critic. Those can be downloaded to watch a "smart" agent.

The 'Report.pdf' file explains how the algorithm works and contains different training examples.

### Resources
- [BATCH NORMALIZATION](https://arxiv.org/abs/1502.03167)
- [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/abs/1509.02971)
- [PARAMETER SPACE NOISE FOR EXPLORATION](https://arxiv.org/pdf/1706.01905.pdf)
