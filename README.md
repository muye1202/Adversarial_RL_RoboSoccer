# Adversarial RL robot soccer
Author: Muye Jia

* under development, this README serves as a debug log and updates.

## Debugg Log
1. The play function in agent.py is used, __think__loop is modified so move to formation message is sent only when the player is not at starting position.
2. The robot player can now move to formation position as designated before kick-off: this is done by putting the commands in a queue, but no multi-threading is used.
3. (For step function) Adding in planning logic so the player would perform meaningful tasks: the player would turn to look for the ball, then walks up to it.

### ROS simulator
1. the direction of the robot can be set immediately to ball direction for simplicity