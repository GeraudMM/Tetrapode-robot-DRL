[//]: # (Image References)

[image1]: walking.gif "Trained Agent"
[image2]: couverture.png "Tetrapode"
[image3]: SlawRealWalk.gif "SlawRealWalk"
[image4]: FastRealWalk.gif "FastRealWalk"



# Tetrapode-robot-learning-to-walk-thanks-to-Deep-Reinforcement-Learning
In this project, the goal is to teach the robot to walk in a Unity simulation using the PPO (Proximal Policy Optimization) algorithm for Deep Reinforcement Learning.

### Introduction
This project follows a school project where we had to design and build a four-legged robot capable of walking in a straight line. This robot consists of a 30 cm long body and four 30 cm arms, each containing three servomotors. To do this, we designed a model of each required part using Catia software, which we then printed in 3D. Once these parts were assembled with the 12 dynamixel motors, we programmed them using an OpenCM card that is similar to an Arduino card. The programming of the walk was done by following two different solutions. The first consists of a polynomial interpolation of the motor angles according to the position of the tool marks (at the end of the arm). The second is to create a simulation of the robot using Matlab's Robotics Toolbox module.

#### With Polynomial Interpolation
![Tetrapode Robot][image3] 

#### With the simulation on Matlab
![Tetrapode Robot][image4] 

### Deep Reinforcement learning
Now, in this project, I tried a third solution which is called [Deep Reinforcement Learning](https://en.wikipedia.org/wiki/Deep_reinforcement_learning). Basically, it gives the agent the ability to distinguish which combination of actions seems to lead to good results from those that lead to bad.
To begin, I have modified the [crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) environment  to create my own environment where I could train the tetrapode robot. This environments are made on Unity using the [ml-agents](https://github.com/Unity-Technologies/ml-agents/) provided by Unity-Technologies.
Then, I began to train the robot using the [DDPG](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)(Deep Deterministic Policy Gradient) algorithm as I used in the [continuous control project](https://github.com/GeraudMM/Continuous-Control-with-DeepRL/blob/master/README.md) from Udacity but it wasn't working very well so I tried the [PPO algorithm](https://github.com/ZeratuuLL/Reinforcement-Learning/blob/master/Continuous%20Control/Crawler/Crawler.ipynb) from [Lifeng Wei](https://github.com/ZeratuuLL) and it directly leds to better results. Then I made some slight changes to the environment and algorithm to get better results. Finally, it manages to walk effectively in a straight line for now. Latter it shouldn't be harder to teach it to walk in whatever direction we choose. [See what can be done].

#### With the PPO algorithm on a Unity environment
![trained Agent][image1]

### Issues and possible solutions
The problem here, though, is that the physic of a tetrapod robot is quit hard and I found that the simulation of Unity couldn't led me to expoitable results for implementation in real life. 

To solve this problem, I can think to three different solutions:
 - At first, we could just try to change the software to a more complex one when talking about physics as OpenAI used for their spider
 - Or, we could try to give robustesse(non anglais) to the robot by training its neural networks on various slightly differentes physic reality like [OpenAI](https://openai.com/) did with [this hand](https://openai.com/blog/learning-dexterity/).
 - Finally, we could directly train the neural network in real life like [here](https://www.youtube.com/watch?v=V05SuCSRAtg). Though it must require the minimum of human intervention so that we can let it train almost alone.

Finally, the simulation is interesteing because even if it's not physicly perfect, it can show us various way to program an efficient walk for the robot.

### Environment
In this environment, a reward of +0.1 is provided for each step that the agent's hand is in the goal location and a reward of +0.03 is provided when the body velocity is in the goal direction. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

This simulation contains 12 tetrapode learning to walk side by side. They all share the same and only brain.

Then, the observation space for the brain consist of 185 variables corresponding to the position, rotation, velocity and angular velocities of each arms and of its own body in addition to the coordinate of where we ask it to go. (We could certainly reduce the number of variables without having a poorer model, but I haven't taken the time to do so yet.) so if we were to implement that in real life we would have to add sensors on the robots. At least gyroscopes and pressure sensors.

Each action is a vector with 24 numbers, corresponding to torque applicable to the 12 joints. Every entry in the action vector should be a number between -1 and 1.


### Files
The `model.py` and `PPO_agent.py` files are containing the class for the Neural Networks and for the Agent.

The `Tetrapode-PPO.ipynb` notebook is the part where we can train and watch a "smart" agent evolving in the Unity environment.

The `MYTETRAPODE_Checkpoint.pth` file is a saving of the weights of the locals actor and critic. Those can be downloaded to watch a "smart" agent.

The 'Report.pdf' file explains how the algorithm works and contains different training examples.

### Resources
- [BATCH Normalization](https://arxiv.org/abs/1502.03167)
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- [Parameter Space Noise For Exploration](https://arxiv.org/pdf/1706.01905.pdf)
- [Transfer from Simulation to Real World through Learning Deep Inverse Dynamics Model](https://arxiv.org/abs/1610.03518)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
