import rclpy
import math
import sys
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Empty
from enum import Enum, auto
import numpy as np
import tensorflow as tf
from rl.ddpg_robo_soccer import OUActionNoise, Buffer, get_actor
from rl.ddpg_robo_soccer import policy, get_critic, update_target
from rl_interfaces.msg import Info

"""
Training loop for the RL network.
"""
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

total_episodes = 10000
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()


class rs_simulator(Node):
    """Publish a robot position and ball marker position"""
    
    def __init__(self):
        super().__init__("rs_simulator")
        
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_training = True
        
        # send kick cmd
        self.kick_pub = self.create_publisher(Point, "field/kick", 10)
        
        # send robot velocity cmd
        self.vel_pub = self.create_publisher(Pose2D, "field/player_vel", 10)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Empty, "~/reset_flag", 10)
        
        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "field/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "field/robot_pos", self.robot_callback, 10)
        
        # receive rewards, states, done variables
        self.rewards_sub = self.create_subscription(Info, "~/step_info", self.step_callback, 10)
        
        # field params
        self.ball_pos = Point()
        self.robot_pos = Pose2D()
        self.robo_state = State.IDLE
        
        # info returned by step function
        self.step_info = Info()
        
        self.recv_update = False
        self.ep = 0.
        self.done_episode = False
        self.prev_state = np.array([0., 0.])
        self.episodic_reward = 0
    
    def step_callback(self, step_info: Info):

        self.step_info = step_info
        self.recv_update = True
    
    def ball_callback(self, ball_pos: Point):
        self.ball_pos = ball_pos
        
    def robot_callback(self, robo_pos: Pose2D):
        self.robot_pos = robo_pos
        
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos.x - self.robot_pos.x)**2 + 
                         (self.ball_pos.y - self.robot_pos.y)**2)
    
    def dash(self, power, dir):
        dash_cmd = Pose2D()
        dash_cmd.x = float(power)
        dash_cmd.y = float(dir)
        self.vel_pub.publish(dash_cmd)
        
    def kick(self, power, dir):
        kick_cmd = Point()
        kick_cmd.x = float(power)
        kick_cmd.y = float(dir)
        self.kick_pub.publish(kick_cmd)
        
    def turning_angle(self, ball: Point, robot: Pose2D):
        # the angle is 0 to PI and -PI to 0
        dely = ball.y - robot.y
        delx = ball.x - robot.x
        ball_to_robot = math.degrees(math.atan2(dely, delx))
        
        a = ball_to_robot
            
        return a
        
    def follow_ball(self, ball_pos: Point):
        # dash towards the ball
        dash_dir = self.turning_angle(ball_pos, self.robot_pos)
        dash_pow = 70
        
        #self.get_logger().info("dash direction: " + str(dash_dir))
        self.dash(dash_pow, dash_dir)
    
    def timer_callback(self):
        # # TEST PASSED: robot dashing forward
        # #       but not kicking the ball
        # #       pose2D: [power, dir, 0.]
        # dash_cmd = Pose2D()
        # dash_cmd.x = 10.
        # dash_cmd.y = 30.
        # self.vel_pub.publish(dash_cmd)
        
        # # TEST PASSED: kick the ball and run to it
        # if self.player_to_ball_dist() <= 0.1:
        #     self.robo_state = State.BALL_KICKABLE

        # if self.robo_state == State.BALL_KICKABLE:
        #     kick_dir = np.random.uniform(low=-55, high=55)
        #     self.kick(70., kick_dir)
        #     self.robo_state = State.IDLE
        
        # self.follow_ball(self.ball_pos)

        if not self.done_episode:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)
            action = policy(tf_prev_state, ou_noise)
            self.get_logger().info("kick pow: " + str(action[0]))

            ###### SENDING THE COMMANDS ######  
            if self.player_to_ball_dist() <= 0.1:
                self.robo_state = State.BALL_KICKABLE

            if self.robo_state == State.BALL_KICKABLE:
                kick_dir = action[1]
                kick_pow = action[0]
                self.kick(kick_pow, kick_dir)
                self.robo_state = State.IDLE
            
            self.follow_ball(self.ball_pos)

            if self.recv_update:
                ###### RECEIVE UPDATES #######
                state = np.array([self.step_info.states])
                reward = self.step_info.rewards
                done = self.step_info.done

                buffer.record((self.prev_state, action, reward, state))
                self.episodic_reward += reward

                buffer.learn()
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    self.done_episode = True
                
                self.prev_state = state
                self.recv_update = False

            ep_reward_list.append(self.episodic_reward)
        
        else:
            self.ep += 1
            self.prev_state = np.array([0., 0.])
            self.episodic_reward = 0
            self.done_episode = False
            
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            self.get_logger().info("Episode * {} * Avg Reward is ==> {}".format(self.ep, avg_reward))
            avg_reward_list.append(avg_reward)


def main(args=None):
    rclpy.init(args=args)
    robot_pub = rs_simulator()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()