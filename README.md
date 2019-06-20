[//]: # (Image References)

[image1]: walking.gif "Trained Agent"
[image2]: couverture.png "Tetrapode"



# Tetrapode-robot-learning-to-walk-thanks-to-Deep-Reinforcement-Learning
In this project I try to learn to the robot to walk in a Unity simulation using the PPO(Proximal Policy Optimization) algorithm for Deep Reinforcement Learning

![trained Agent][image1]

This project follow a school project where we had to design and build a quadripod robot able of walking in a straight line. This robot will consist of a 30 cm long body and four 30 cm arms  containing three servomotors each. To do this, we have design a model of each necessary part using the Catia software that we have then print in 3D. Once these parts have been assembled to the 12 dynamixel motors, we have done the programation using an OpenCM card which is similar to an Arduino card. The programming of the operation has be done by following two different solutions. The first one consists of a polynomial interpolation of the motor angles according to the position of the tool marks (at the end of the arm). The second involves to create a simulation of the robot using the Robotics Toolbox module on Matlab.

![Tetrapode Robot][image2] 

Now, in this project, I have at first modified the reacher environment (link) to create my own environment for training the tetrapode then, I tried to train the robot using the DDPG algorithm but it wasn't working very well so I tried the PPO algorithm from (citer le gars a qui j'ai pris l'algorithme) and it led to better results. Then I modified the environment and the algorithm so that it works well. The problem here, though, is that the physic of a tetrapod robot is quit hard and the simulation of Unity isn't satysfying. It seems to be too far from reality to lead to good results in real life. 
To solve this problem, I can think to three different solutions:
- At first, we could just try to change the software to a more complex one when talking about physics
-or, we could try to give robustesse(non anglais) to the robot by training its neural networks on various slightly differentes physic reality( link to the page of OpenAI sur le bras robotique)
-Finally, we could directly train the neural network in real life. The problem is that it could needs human intervention and take many many time (links to the robots learning IRL, many links if needed)

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
