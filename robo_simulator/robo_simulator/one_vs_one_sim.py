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
from rl.ddpg_robo_soccer import OUActionNoise, DDPG_robo
from rl_interfaces.msg import Info
import matplotlib.pyplot as plt
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


std_dev = 0.05
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Used to update target networks
tau = 0.002
gamma = 0.99

total_episodes = 4000

"""
Attacker network
"""

# attacker_ddpg = DDPG_robo(first_low=30., first_high=70., sec_low=-90, sec_high=90., flag="")
# # To store reward history of each episode
# ep_reward_list = []
# # To store average reward history of last few episodes
# avg_reward_list = []

"""
Defender network
"""
defender_ddpg = DDPG_robo(first_low=40., first_high=70., sec_low=-100, sec_high=100, num_states=8, flag="defender")
# defender_ddpg.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_actor_4000.h5")
# defender_ddpg.critic_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_critic_checkpoint.h5")
# defender_ddpg.target_actor.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_target_actor_checkpoint.h5")
# defender_ddpg.target_critic.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_target_critic_checkpoint.h5")
defender_ep_rewards = []
defender_avg_rewards = []

kickpow_upper_bound = 70.
kickpow_lower_bound = 40.
kickdir_low = -90.
kickdir_high = 90.


class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()


class one_one_simulator(Node):
    """Simulate one vs one scenario"""
    
    def __init__(self):
        super().__init__("one_vs_one")

        self.cb_group = MutuallyExclusiveCallbackGroup()
        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback, callback_group=self.cb_group)
        self.start_training = True

        # send kick cmd
        self.kick_pub = self.create_publisher(Point, "one_one/kick", 10)

        # send robot velocity cmd
        self.vel_pub = self.create_publisher(Pose2D, "one_one/player_vel", 10)
        # send defender dash cmd
        self.defender_vel_pub = self.create_publisher(Pose2D, "one_one/defender_vel", 10)

        # step update pub for defender
        self.defender_state_pub = self.create_publisher(Empty, "one_one/defender_update", 10)
        self.attacker_update_pub = self.create_publisher(Empty, "one_one/update", 10)

        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "one_one/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "one_one/robot_pos", self.robot_callback, 10)
        
        # receive rewards, states, done variables
        self.defender_rewards_sub = self.create_subscription(Info, "defend/defender_info", self.defender_step_callback, 1)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Pose2D, "one_one/reset_flag", 10)
        
        # field params
        self.ball_pos = Point()
        self.robot_pos = Pose2D()
        self.robo_state = State.IDLE
        
        # info returned by step function
        self.step_info = Info()
        self.recv_update = False   # tell rs_env if data should be analyzed
        self.ep = 0.
        self.done_episode = False
        self.prev_state = np.array([0., 0.])
        self.episodic_reward = 0.0
        self.start_train = True
        self.action_list = []
        
        # DEFENDER params
        self.defender_prev_state = np.array([2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.2, 0.0])
        self.defender_ep_rewards = 0.0
        self.defender_pos = Pose2D()
        self.defender_info = None
        self.defender_episodic_rewards = 0.0
        self.defender_recv_update = False
        
        self.ball_prev_pos = np.array([0.2, 0.0])
        self.defender_prev_pos = np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.2, 0.0])
        self.arena_range_x = 3.5
        self.arena_range_y = self.arena_range_x
        self.flag = False
        self.new_pos = Pose2D()
        
        # training log
        # self.train_log = open("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/train_log/rewards_log.txt", "a")
        # self.defender_action_log = open("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/train_log/defender_action_log.txt", "a")
        
        # load attacker model
        self.actor_model = DDPG_robo(0., 0., 0., 0., num_states=2, flag="predict")
        self.actor_model.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_attacker/attacker_actor.h5")

    def defender_step_callback(self, step_info: Info):
        
        self.defender_info = step_info
        self.defender_recv_update = True

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

    def defender_dash(self, dash_speed, dash_dir):
        dash_cmd = Pose2D()
        dash_cmd.x = float(dash_speed)
        dash_cmd.y = float(dash_dir)
        self.defender_vel_pub.publish(dash_cmd)

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
        dash_pow = 60.

        self.dash(dash_pow, dash_dir)

    def reset_signal(self):
        """
        Send the reset signal to simulator to reset the position
        of the robot and the ball.
        """
        self.new_pos.x = -2.0
        self.new_pos.y = np.random.uniform(low=-3.0, high=3.0)
        self.reset_pub.publish(self.new_pos)

    def is_scored(self):
        """
        Check whether the attacking side has scored.
        """
        
        return (self.ball_pos.x >= 0.5*self.arena_range_x-0.2 and
                self.ball_pos.y <= 0.1*self.arena_range_y-0.2 and
                self.ball_pos.y >= 0.1*self.arena_range_y+0.2)
        
    def is_dead_ball(self):
        """
        Check whether the ball is out of range.
        """
        
        return (self.ball_pos.x <= -self.arena_range_x-0.2 or
                self.ball_pos.x >= self.arena_range_x+0.2 or
                self.ball_pos.y <= -self.arena_range_y-0.2 or
                self.ball_pos.y >= self.arena_range_y+0.2)

    def angle_between(self):
        """
        Angle between players.
        """
        
        return math.degrees(math.atan2(self.robot_pos.y - self.defender_pos.y,
                                       self.robot_pos.x - self.defender_pos.x))

    def player_facing(self):
        """
        Is player facing the target (compare angles).
        """
        angle_between_players = self.angle_between()
        def_facing = math.degrees(self.defender_pos.theta)

        if def_facing < 0:
            def_facing = 180 - abs(def_facing)
            
        elif def_facing >= 0:
            def_facing = 180 + def_facing
            
        if angle_between_players < 0:
            angle_between_players = 360 - abs(angle_between_players)
            
        angle_diff = angle_between_players - def_facing
        return angle_diff

    def timer_callback(self):
        if self.ep <= total_episodes:
            if not self.done_episode:
                self.flag = True
                ####### ATTACKER TRAINING LOOP ######
                #####################################
                ######## SENDING THE COMMANDS #######
                ball_pos = np.array([self.ball_pos.x, self.ball_pos.y])
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(ball_pos), 0)
                action = self.actor_model.actor_model.predict(tf_prev_state, verbose=0)
                outputs = action[0] * kickdir_high
                outputs[0] = tf.clip_by_value(outputs[0], kickpow_lower_bound, kickpow_upper_bound)   # kick power
                outputs[1] = tf.clip_by_value(outputs[1], kickdir_low, kickdir_high) # kick direction

                if self.player_to_ball_dist() <= 0.1:
                    self.robo_state = State.BALL_KICKABLE

                if self.robo_state == State.BALL_KICKABLE:
                    kick_dir = outputs[1]
                    kick_pow = outputs[0]
                    self.kick(kick_pow, kick_dir)

                    self.attacker_update_pub.publish(Empty())
                    self.robo_state = State.IDLE

                self.follow_ball(self.ball_pos)

                ###### DEFENDER TRAINING LOOP #######
                #####################################
                # sample actions from the network
                # for defender, take ball pos into input
                defender_prev_state = tf.expand_dims(tf.convert_to_tensor(self.defender_prev_state), 0)
                defender_action = defender_ddpg.policy(defender_prev_state, noise_object=ou_noise)
                dash_speed = defender_action[0]
                dash_dir = defender_action[1]

                self.defender_dash(dash_speed, dash_dir)
                self.defender_state_pub.publish(Empty())
                if self.defender_recv_update:
                    state_list = self.defender_info.states
                    state = np.array([state_list[0], state_list[1], state_list[2], state_list[3], state_list[4],
                                      state_list[5], self.ball_pos.x, self.ball_pos.y])
                    def_reward = self.defender_info.rewards
                    def_done = self.defender_info.done

                    # normalize the states input
                    normal_state = state
                    normal_pre_state = self.defender_prev_state

                    if np.linalg.norm(state) > 0 and np.linalg.norm(self.defender_prev_state) > 0:
                        normal_state = state / np.linalg.norm(state)
                        normal_pre_state = self.defender_prev_state / np.linalg.norm(self.defender_prev_state)

                    defender_ddpg.buffer.record(obs_tuple=(normal_pre_state, defender_action, def_reward, normal_state))
                    self.defender_episodic_rewards += def_reward

                    defender_ddpg.buffer.learn(defender_ddpg.target_actor, defender_ddpg.target_critic,
                                               defender_ddpg.actor_model, defender_ddpg.critic_model, 
                                               defender_ddpg.critic_optimizer, defender_ddpg.actor_optimizer, gamma=gamma)
                    defender_ddpg.update_actor_target(tau)
                    defender_ddpg.update_critic_target(tau)

                    # End this episode when `done` is True
                    if def_done:
                        self.done_episode = True
                        self.reset_signal()

                    self.defender_prev_state = normal_state
                    self.defender_prev_pos = state
                    self.defender_recv_update = False

                    defender_ep_rewards.append(self.defender_episodic_rewards)

            elif self.done_episode and self.flag:
                self.flag = False
                self.ep += 1
                
                # reset the state input of the defender
                angle = math.degrees(math.atan2(self.new_pos.y - 0.0,
                                                self.new_pos.x - 2.0))
                dist_between = math.sqrt((self.new_pos.x - 2.0)**2 + (self.new_pos.y)**2)
                self.defender_prev_state = np.array([2., 0., 0.0, dist_between, angle,
                                                     angle, self.new_pos.x+0.2, self.new_pos.y])
                self.defender_prev_state = self.defender_prev_state / np.linalg.norm(self.defender_prev_state)

                self.episodic_reward = 0
                self.defender_episodic_rewards = 0
                self.done_episode = False
                
                def_avg_reward = np.mean(defender_ep_rewards[-40:])
                
                # training log
                # self.train_log.write("Episode * {} * DEFENDER Avg Reward is ==> {} \n".format(self.ep, def_avg_reward))
                self.get_logger().info("Episode * {} * DEFENDER Avg Reward is ==> {}".format(self.ep, def_avg_reward))
                defender_avg_rewards.append(def_avg_reward)

                # save the weights every 1000 episodes:
                if self.ep % 100 == 0:
                   defender_ddpg.actor_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_actor_checkpoint.h5")
                   defender_ddpg.critic_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_critic_checkpoint.h5")

                   defender_ddpg.target_actor.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_target_actor_checkpoint.h5")
                   defender_ddpg.target_critic.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_target_critic_checkpoint.h5")

        else:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(defender_avg_rewards, label="defender")
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()
            
            defender_ddpg.actor_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_actor_4000.h5")
            # defender_ddpg.critic_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_8000_critic.h5")

            # defender_ddpg.target_actor.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_8000_target_actor.h5")
            # defender_ddpg.target_critic.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_8000_target_critic.h5")


def main(args=None):
    rclpy.init(args=args)
    robot_pub = one_one_simulator()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
    

######### TRAINING ATTACKER FROM SCRATCH ############
    
# state_list = self.step_info.states
# state = np.array([state_list[0], state_list[1]])
# reward = self.step_info.rewards
# done = self.step_info.done

# normal_state = state
# normal_pre_state = self.prev_state

# # normalize input
# if np.linalg.norm(state) > 0 and np.linalg.norm(self.prev_state) > 0:
#     normal_state = state / np.linalg.norm(state)
#     normal_pre_state = self.prev_state / np.linalg.norm(self.prev_state)

# attacker_ddpg.buffer.record(obs_tuple=(normal_pre_state, action, reward, normal_state))
# self.episodic_reward += reward

# attacker_ddpg.buffer.learn(attacker_ddpg.target_actor, attacker_ddpg.target_critic,
#                             attacker_ddpg.actor_model, attacker_ddpg.critic_model, 
#                             attacker_ddpg.critic_optimizer, attacker_ddpg.actor_optimizer, gamma=gamma)
# attacker_ddpg.update_actor_target(tau)
# attacker_ddpg.update_critic_target(tau)

# # End this episode when `done` is True
# if done:
#     self.done_episode = True
#     self.reset_signal()

# self.prev_state = normal_state
# self.ball_prev_pos = state
# self.recv_update = False

# ep_reward_list.append(self.episodic_reward)
# defender_ep_rewards.append(self.defender_episodic_rewards)