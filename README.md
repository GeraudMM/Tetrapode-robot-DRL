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

