[//]: # (Image References)

[image1]: walking.gif "Trained Agent"
[image2]: couverture.png "Tetrapode"



# Tetrapode-robot-learning-to-walk-thanks-to-Deep-Reinforcement-Learning
In this project we learn to the robot to walk in a Unity simulation using the PPO(Proximal Policy Optimization) algorithm for Deep Reinforcement Learning

![trained Agent][image1]

This project follow a school project where we had to design and build a quadripod robot able of walking in a straight line. This robot will consist of a 30 cm long body and four 30 cm arms  containing three servomotors each. To do this, we have design a model of each necessary part using the Catia software that we have then print in 3D. Once these parts have been assembled to the 12 dynamixel motors, we have do the programation using an OpenCM card which is similar to an Arduino card. The programming of the operation has be done by following two different solutions. The first one consists of a polynomial interpolation of the motor angles according to the position of the tool marks (at the end of the arm). The second involves to create a simulation of the robot using the Robotics Toolbox module on Matlab.

![Tetrapode Robot][image2]
