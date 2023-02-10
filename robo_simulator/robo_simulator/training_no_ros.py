import math
import sys
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from enum import Enum, auto
from rl.ddpg_robo_soccer import OUActionNoise, Buffer, get_actor
from rl.ddpg_robo_soccer import policy, get_critic, update_target


"""
Training loop for the RL network.
"""
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
#actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/successful_model/one_attacker/attacker_actor_2000.h5")
critic_model = get_critic()
#critic_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/successful_model/one_attacker/attacker_critic_2000.h5")

target_actor = get_actor()
#target_actor.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/successful_model/one_attacker/attacker_target_actor_2000.h5")
target_critic = get_critic()
#target_critic.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/successful_model/one_attacker/attacker_target_critic_2000.h5")

total_episodes = 1000
# Used to update target networks
tau = 0.005
gamma = 0.99
buffer = Buffer(50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()
    DRIBBLING = auto()
    KICKING = auto()
    DASHING = auto()
    STOPPED = auto()


class RoboEnv():
    
    arena_range_x = 3.5
    arena_range_y = arena_range_x/2
    
    def __init__(self):
        self.robot_facing = 0.   # should be angles
        self.robot_pos = np.array([0., 0.])
        self.ball_pos = np.array([0., 0.])

        # last ball to goal dist
        self.last_ball_dist = self.arena_range_x
        
    def step(self, robot_facing, robot_pos, ball_pos):
        done = False
        rewards = 0.0
        
        self.robot_facing = robot_facing  # should be degrees
        self.robot_pos = robot_pos
        self.ball_pos = ball_pos

        ### REWARDS ###
        # For attackers:
        #       1. For each step without scoring, reward = 0
        #       2. Lose the control of the ball, reward = -10
        #       3. Shooting route blocked by defender, reward = -1
        #       4. Find a clear route to goal, reward = +5
        #       5. Score a goal, reward is set to 10
        #       6. The ball's distance towards the goal advances, reward = + 0.5
        #       7. The ball is out of field, reward = -5
        #       8. Inside shooting range to the goal, reward = + 2
        # For defenders:
        #       For each step without being scored, reward = +1
        #       Taking control over the ball, reward = +10
        #       Blocking shooting route, reward = +1
        
        # check whether the robot moves towards
        # the goal in the past episode:
        if self.last_ball_dist - self.ball_to_goal_dist() > 0.01:    
            rewards += 0.5
            
            # self.get_logger().info("to goal dist decreased: " + str(self.ball_to_goal_dist()-self.last_ball_dist))
            # self.last_ball_dist = self.ball_to_goal_dist()

        elif self.last_ball_dist - self.ball_to_goal_dist() < 0.01:
            rewards -= 0.1
            
            # self.get_logger().info("to goal dist increased: " + str(self.ball_to_goal_dist()-self.last_ball_dist))
            # self.last_ball_dist = self.ball_to_goal_dist()
            
        #The robot should be rewarded if it faces the goal direction
        if self.is_player_facing_goal():
            rewards += 0.1
            
        # check whether the robot is advancing to the goal
        if self.is_scored():
            rewards += 10.
            done = True
            
        if not self.is_scored() and self.is_dead_ball():
            if rewards == 0.:
                rewards = -0.01
            else:
                rewards -= 0.01

            done = True
        
        states = [self.ball_pos[0], self.ball_pos[1]]
        self.last_ball_dist = self.ball_to_goal_dist()
        return states, rewards, done
            
    def is_scored(self):
        """
        Check whether the attacking side has scored.
        """
        
        return (self.ball_pos[0] >= 0.5*self.arena_range_x-0.2 and
                self.ball_pos[1] <= 0.1*self.arena_range_y-0.2 and
                self.ball_pos[1] >= 0.1*self.arena_range_y+0.2)
        
    def is_dead_ball(self):
        """
        Check whether the ball is out of range.
        """
        
        return (self.ball_pos[0] <= -self.arena_range_x-0.2 or
                self.ball_pos[0] >= self.arena_range_x+0.2 or
                self.ball_pos[1] <= -self.arena_range_y-0.2 or
                self.ball_pos[1] >= self.arena_range_y+0.2)
    
    def ball_to_goal_dist(self):
        """
        Calculate the distance between the ball and the goal
        """
        
        return (self.arena_range_x - self.ball_pos[0])**2 + (self.ball_pos[1])**2
    
    def is_player_facing_goal(self):
        """
        Determine if the robot is facing the goal
        """
        
        # calculate the angle between goal's left post
        # and the robot
        left_post_angle = math.degrees(math.atan2(0.4*2*self.arena_range_y-self.robot_pos[1], 
                                                  self.arena_range_x-self.robot_pos[0]))
        
        right_post_angle = math.degrees(math.atan2(self.robot_pos[1]+0.4*2*self.arena_range_y, 
                                                  self.arena_range_x-self.robot_pos[0]))
        robot_facing = self.robot_facing
        
        return (robot_facing <= left_post_angle and
                robot_facing >= right_post_angle)
        
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos[0] - self.robot_pos[0])**2 + 
                         (self.ball_pos[1] - self.robot_pos[1])**2)

class Train():

    def __init__(self):
        self.robot_pos = np.array([0., 0.])
        self.ball_pos = np.array([0.2, 0.0])
        self.MAX_KICK_VEL = 30
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.4
        self.PLAYER_MAX_SPEED = 1.0
        self.TIME_STEP = 0.1  # simulator timestep 1.0 second
        self.last_vel = 0.
        self.kick_power = 0.0
        self.kick_dir = 0.0
        self.ang = 0.0
        self.velx = 0.
        self.vely = 0.
        self.robo_state = State.IDLE
        self.player_state = State.STOPPED
        self.ball_state = State.STOPPED
        self.env = RoboEnv()
        self.prev_ball_pos = np.array([0.2, 0.0])
        
        # store training states for visualization
        self.ball_replay_x = []
        self.ball_replay_y = []
        self.x_trajectories = []
        self.y_trajectories = []


    def player_to_ball_dist(self):
            
        return math.sqrt((self.ball_pos[0] - self.robot_pos[0])**2 + 
                            (self.ball_pos[1] - self.robot_pos[1])**2)

    def kick_update(self, point):
        """kick: [pow, dir]"""
        self.kick_power = point[0]
        self.kick_dir = (point[1] * math.pi)/180.
        self.ang = math.degrees(self.kick_dir)
        
        # when receiving new effective kick
        # the ball will move the farthest this time
        self.last_vel = (self.kick_power/self.MAX_STRENGTH)*self.MAX_KICK_VEL
        
        self.ball_state = State.KICKING

    def calc_ball_pos(self):
        """Calculate ball position at each time step"""
        # update the ball position
        new_x = self.last_vel*self.TIME_STEP*math.cos(self.kick_dir)
        new_y = self.last_vel*self.TIME_STEP*math.sin(self.kick_dir)

        self.ball_pos[0] += new_x
        self.ball_pos[1] += new_y

        # the ball moves farthest at first kick
        # then decays to zero.
        # calc new moving dist from last moving dist.
        new_vel = self.last_vel*self.BALL_DECAY
        self.last_vel = new_vel
        
    def turning_angle(self, ball, robot):
        # the angle is 0 to PI and -PI to 0
        dely = ball[1] - robot[1]
        delx = ball[0] - robot[0]
        ball_to_robot = math.degrees(math.atan2(dely, delx))

        a = ball_to_robot
            
        return a
        
    def follow_ball(self, ball_pos):
        # dash towards the ball
        dash_dir = self.turning_angle(ball_pos, self.robot_pos)
        dash_dir = (dash_dir * np.pi)/180.
        dash_pow = 80.
        self.ang = math.degrees(dash_dir)

        dash_speed = (dash_pow/100)*self.PLAYER_MAX_SPEED
        self.velx = dash_speed * math.cos(dash_dir)
        self.vely = dash_speed * math.sin(dash_dir)

        # update the player as dashing
        self.player_state = State.DASHING

    def training_loop(self):

        for episodes in range(total_episodes):
            prev_state = np.array([0.2, 0.])
            self.ball_pos = np.array([0.2, 0.0])
            self.prev_ball_pos = np.array([0.2, 0.0])
            self.robot_pos = np.array([0., 0.])
            self.ang = 0.0
            episodic_rewards = 0.0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_ball_pos), 0)
                action = policy(tf_prev_state, actor_model=actor_model, noise_object=ou_noise)

                ###### UPDATE STATES #######
                self.calc_ball_pos()
                if self.player_to_ball_dist() <= 0.1:
                    self.robo_state = State.BALL_KICKABLE

                if self.robo_state == State.BALL_KICKABLE:
                    kick_dir = action[1]
                    kick_pow = action[0]
                    self.kick_update((kick_pow, kick_dir))

                    self.robo_state = State.IDLE

                self.follow_ball(self.ball_pos)

                ball_to_player = self.player_to_ball_dist()
                if (self.player_state == State.DASHING and 
                   not self.ball_state == State.KICKING and 
                   ball_to_player <= 0.1):
                    rb_facing = (self.ang * np.pi)/180.
                    self.ball_pos[0] = self.robot_pos[0] + ball_to_player*math.cos(rb_facing)
                    self.ball_pos[1] = self.robot_pos[1] + ball_to_player*math.sin(rb_facing)
                    
                elif ball_to_player > 0.1:
                    self.robot_pos[0] += self.TIME_STEP * self.velx
                    self.robot_pos[1] += self.TIME_STEP * self.vely

                ###### GET REWARDS ######
                state, reward, done = self.env.step(robot_facing=self.ang, robot_pos=self.robot_pos,
                              ball_pos=self.ball_pos)

                episodic_rewards += reward

                normal_state = state
                normal_pre_state = prev_state
                # normalize input
                if np.linalg.norm(state) > 0 and np.linalg.norm(prev_state) > 0:
                    normal_state = state / np.linalg.norm(state)
                    normal_pre_state = prev_state / np.linalg.norm(prev_state)

                buffer.record((normal_pre_state, action, reward, normal_state))
                episodic_rewards += reward

                buffer.learn(target_actor=target_actor, target_critic=target_critic,
                             actor_model=actor_model, critic_model=critic_model, gamma=gamma)
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

                if done:
                    break

                prev_state = normal_state
                self.prev_ball_pos = state
                self.player_state = State.STOPPED
                self.ball_state = State.STOPPED

                if episodes % 200 == 0:
                    self.ball_replay_x.append(self.ball_pos[0])
                    self.ball_replay_y.append(self.ball_pos[1])

            ep_reward_list.append(episodic_rewards)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(episodes, avg_reward))
            avg_reward_list.append(avg_reward)

            if episodes % 200 == 0:
                self.x_trajectories.append(self.ball_replay_x)
                self.y_trajectories.append(self.ball_replay_y)

            # save the weights every 1000 episodes:
            if episodes % 1000 == 0:
                actor_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/test1_actor.h5")
                critic_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/test1_critic.h5")

                target_actor.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/test1_target_actor.h5")
                target_critic.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/test1_target_critic.h5")

        # plot ball trajectories
        for i in range(len(self.x_trajectories)):
            plt.plot(self.x_trajectories[i], self.y_trajectories[i], label=(str((i+1)*200) + "th trajectory"))

        plt.legend()
        plt.xlabel("ball x position")
        plt.ylabel("ball y position")
        plt.show()
            
                
if __name__ == "__main__":
    train_loop = Train()
    train_loop.training_loop()