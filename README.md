# Adversarial RL robot soccer
Author: Muye Jia

Training two teams of robot to play soccer and improve by training against each other.

## Demo
The following is the defending robot (red) trying to defend the attacking robot (blue) so it does not dribble the ball to the goal.



https://user-images.githubusercontent.com/112987403/226082364-88d2fe2f-e15e-4aae-9427-3c9ce6c3eaf2.mp4

The second demo is the naive attacker robot trying to take the ball to the opening (goal).



https://user-images.githubusercontent.com/112987403/226082654-69e730d0-4654-4455-b459-20fa8ac2c169.mp4


## Behind the Scenes
Both the attacker and the defender is trained using Deep Deterministic Policy Gradient (DDPG) algorithm. For the 1 vs 1 scenario, the defender knows the position of the attacking robot, its heading, and its the angular difference between the attacker's heading and its own heading-these parameters are included as input to the DDPG network, and the output of the network is the velocity (speed and direction) of the defending robot.

### Neural Network Details
The defender is trained using a 5-layer fully-connected neural network, the first three layers used leaky-relu as the activation function since it accounts for the negative values in the input data; and the last two layers, plus the output layer, use hyperbolic tangent (tanh) activation function so the output is constrained within -1 to 1, which is then scaled to obtain speed and direction (in degrees).

To alleviate the vanishing and exploding gradient problem induced by the two activation functions of choice, I used Glorot Normal initializer for tanh activation, and He Normal for Leaky-relu activation, which works surprisingly well in basically maintaining a stable gradient throughout the training process.

3-step TD target is implemented instead of the one-step TD target, this allows the update to propagate along the entire trajectory and the estimation more accurate since it makes use of three data points of experience.

## How to Evaluate the Successful Model
1. Build the project workspace by running `colcon build`
2. Go to the `robo_simulator/robo_simulator/defender_evaluation.py` file line 61 to load the weights of actor model for evaluation.
    - The successful model is stored in `successful_model` (models for both 1vs1 scenario and naive attacker)
3. Use `ros2 launch nuturtle_description one_vs_one.launch.xml` to start the simulation environment using Rviz2.
4. Then run `ros2 run robo_simulator defender_eval` to run the 1 vs 1 evaluation node.

## How to run the python training loop
1. Navigate to `train_loop` folder, `att_def_train_local.py` is for training 1 vs 1 scenario, `attacker.py` is for training the attacker taking the ball towards the goal. Modified the following path on top of the training files to match your own directories.

![Screenshot from 2023-03-17 23-28-51](https://user-images.githubusercontent.com/112987403/226084713-7e26a44e-da31-4eb4-8034-4266f4068335.png)
