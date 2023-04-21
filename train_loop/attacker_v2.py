"""
Train the attacker against the aggressive defender.
"""
import math
import sys
import os

save_path = '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent'
if os.path.exists('/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent'):
    sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
elif os.path.exists("/home/muye/Adversarial_RL_RoboSoccer"):
    save_path = "/home/muye/Adversarial_RL_RoboSoccer"
    sys.path.insert(0, "/home/muye/Adversarial_RL_RoboSoccer")
else:
    save_path = '/home/muye/rl_soccer'
    sys.path.insert(0, '/home/muye/rl_soccer')

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from enum import Enum, auto
from rl.ddpg_robo_soccer import OUActionNoise, DDPG_robo
import matplotlib.pyplot as plt


std_dev = 0.05
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

total_episodes = 15000
# Used to update target networks
tau = 0.0005
gamma = 0.9

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

kickpow_upper_bound = 60.
kickpow_lower_bound = 40.
kickdir_low = -90.
kickdir_high = 90.


class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()
    DRIBBLING = auto()
    KICKING = auto()
    DASHING = auto()
    STOPPED = auto()
    
    
class AttackerEnv():
    arena_range_x = 3.5
    arena_range_y = arena_range_x
    
    def __init__(self) -> None:
        self.ball_pos = np.array([0.2, 0.0])
        self.attacker_pos = np.array([0.0, 0.0, 0.0])
        self.defender_pos = np.array([2.0, 0.0, 0.0])
        self.count_steps = 0
    
    def step(self, ball_pos, attacker_pos, defender_pos):
        done = False
        rewards = 0.0
        self.ball_pos = ball_pos
        self.attacker_pos = attacker_pos
        self.defender_pos = defender_pos

        ### DEFENDER REWARDS ###
        # For each step without being scored, reward = +1
        # Taking control over the ball, reward = +10
        # Blocking shooting route, reward = +1
        # check whether the robot moves towards
        # the goal in the past episode:
        if self.is_scored():
            rewards += 3000.0
            done = True

        # stop the defender when it runs pass the attacker
        if not self.is_scored() and self.attacker_out_of_range():
            rewards -= 2000.0
            done = True

        if self.dist_between_players() > 1.6:
            rewards += 100.0 + 5*abs(self.dist_between_players() - 1.6)

        elif self.dist_between_players() <= 0.8:
            done = True
            rewards -= (1000.0 + 100/(0.8 - self.dist_between_players()))

        if self.dist_between_players() <= 1.6 and self.player_facing() <= 15.0:
            rewards -= 5.0 * 15.0 / (self.player_facing() + 1)

        # if the player chases the opponent directly
        # it will obtain the max rewards
        if self.chasing_score() <= 10.0:
            rewards -= 5*10.0 / (self.chasing_score() + 1)

        if self.ball_to_goal_dist() <= 2.5:
            rewards += 100.0 + 10*(2.5 - self.ball_to_goal_dist())

        extra_states = np.array([self.dist_between_players(), self.angle_between()])
        
        return rewards, done, extra_states

    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos[0] - self.attacker_pos[0])**2 + 
                         (self.ball_pos[1] - self.attacker_pos[1])**2)

    def dist_to_start(self):
        """
        Measure the Euclidean dist to start pos.
        """
        
        return abs(self.defender_pos[0] - (-3.5))

    def angle_between(self):
        """
        Angle between players.
        """

        return math.degrees(math.atan2(self.attacker_pos[1] - self.defender_pos[1],
                                       self.attacker_pos[0] - self.defender_pos[0]))

    def player_facing(self):
        """
        Is player facing the target (compare angles).
        """
        angle_between_players = self.angle_between()
        def_facing = math.degrees(self.defender_pos[2])

        if angle_between_players < 0:
            angle_between_players = 360 - abs(angle_between_players)

        return angle_between_players - def_facing

    def dist_between_players(self):
        """
        Calculate distance between attacker and defender
        """
        
        return math.sqrt((self.attacker_pos[0] - self.defender_pos[0])**2
                          + (self.attacker_pos[1] - self.defender_pos[1])**2)

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
        left_post_angle = math.degrees(math.atan2(0.4*2*self.arena_range_y-self.attacker_pos[1],
                                                  self.arena_range_x-self.attacker_pos[0]))

        right_post_angle = math.degrees(math.atan2(self.attacker_pos[1]+0.4*2*self.arena_range_y,
                                                  self.arena_range_x-self.attacker_pos[0]))
        robot_facing = self.attacker_pos[2]

        return (robot_facing <= left_post_angle and
                robot_facing >= right_post_angle)
        
    def attacker_out_of_range(self):
        """
        Determine if the defender is out of field.
        
        Note: the episode will end if the defender run pass the attacker.
        """

        return (self.attacker_pos[0] <= -self.arena_range_x-0.2 or
                self.attacker_pos[0] >= self.arena_range_x+0.2 or
                self.attacker_pos[1] <= -self.arena_range_y-0.2 or
                self.attacker_pos[1] >= self.arena_range_y+0.2)

    def chasing_score(self):
        """
        Check how much does the player advance in the direction of the opponent.
        
        Calculate the dot product between the vector from the player to the opponent,
        and the vector from its last to current position; then assign rewards proportional
        to the dot product
        """
        vect_to_opponent = np.array([self.attacker_pos[0] - 2.0, 
                                     self.attacker_pos[1] - 0.0])
        vect_to_opponent = vect_to_opponent / np.linalg.norm(vect_to_opponent)

        vect_to_curr = np.array([self.defender_pos[0] - 2.0,
                                 self.defender_pos[1] - 0.0])
        vect_to_curr = vect_to_curr / np.linalg.norm(vect_to_curr)

        dot_product = np.dot(vect_to_curr, vect_to_opponent)
        return math.degrees(np.arccos(dot_product))


def attacker_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    model = models.Sequential()
    model.add(tf.keras.Input(shape=(2)))
    model.add(layers.Dense(256, activation="relu", use_bias=True))
    model.add(layers.Dense(256, activation="tanh", use_bias=True))
    model.add(layers.Dense(2, activation="tanh", use_bias=True, kernel_initializer=last_init))
    
    return model

def defender_actor():
    """
    Trained defender actor with aggressive defending behavior.
    """
    leaky_relu = layers.LeakyReLU(alpha=0.3)
    model = models.Sequential()
    model.add(layers.Input(shape=(8)))
    model.add(layers.Dense(256, activation=leaky_relu, use_bias=True))
    model.add(layers.Dense(256, activation=leaky_relu, use_bias=True))
    model.add(layers.Dense(256, activation=leaky_relu, use_bias=True))
    model.add(layers.Dense(256, activation="tanh", use_bias=True))
    model.add(layers.Dense(256, activation="tanh", use_bias=True))
    model.add(layers.Dense(2, activation="tanh", use_bias=True))
    
    return model


class Train():
    def __init__(self):
        # attacker params
        self.attacker_pos = np.array([0., 0., 0.0])
        self.ball_pos = np.array([0.2, 0.0])
        self.MAX_KICK_VEL = 30
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.4
        self.PLAYER_MAX_SPEED = 0.6
        self.TIME_STEP = 0.05  # simulator timestep 1.0 second
        self.last_vel = 0.
        self.kick_power = 0.0
        self.kick_dir = 0.0
        self.attacker_ang = 0.0
        self.velx = 0.
        self.vely = 0.
        self.attacker_state = State.IDLE
        self.player_state = State.STOPPED
        self.ball_state = State.STOPPED
        self.prev_ball_pos = np.array([0.2, 0.0])
        train_log_dir = save_path + "/train_log/local"
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # ATTACKER parameters
        self.env = AttackerEnv()
        self.attacker_actor = DDPG_robo(first_low=30.0, first_high=100.0, sec_low=-140.0, sec_high=140.0, num_states=7, flag="attacker")
        self.extra_states = np.zeros((3,))
        self.att_extra_states = np.array([4.0, 0.0])

        # DEFENDER parameters
        self.defender_actor = defender_actor()
        self.defender_actor.load_weights(save_path + "/successful_model/1vs1/aggressive_feb_27/defender_checkpt_alien.h5", by_name=True)
        self.defender_pos = np.array([2.0, 0.0, 0.0])
        self.DEFENDER_MAX_SPEED = 1.0

        # DEBUG:
        self.rewards_list = []
        self.episodes_axis = []
        self.x_pos = []
        self.y_pos = []
        self.theta_list = []
        self.att_xpos = []
        self.att_ypos = []
        
    def player_to_ball_dist(self):
            
        return math.sqrt((self.ball_pos[0] - self.attacker_pos[0])**2 + 
                            (self.ball_pos[1] - self.attacker_pos[1])**2)

    def kick_update(self, point):
        """kick: [pow, dir]"""
        self.kick_power = point[0]
        self.kick_dir = (point[1] * math.pi)/180.
        self.attacker_ang = math.degrees(self.kick_dir)
        
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
        dash_dir = self.turning_angle(ball_pos, self.attacker_pos)
        dash_dir = (dash_dir * np.pi)/180.
        dash_pow = 50.
        self.attacker_ang = math.degrees(dash_dir)

        dash_speed = (dash_pow/100)*self.PLAYER_MAX_SPEED
        self.velx = dash_speed * math.cos(dash_dir)
        self.vely = dash_speed * math.sin(dash_dir)

        # update the player as dashing
        self.player_state = State.DASHING
        
    def defender_dash(self, def_vel):
        """
        Process defender's dash command
        
        Input:
            - def_vel: [speed, dir, 0]
        """
        self.def_dash_speed = (def_vel[0]/100.) * self.DEFENDER_MAX_SPEED
        self.def_dash_dir = (def_vel[1] * math.pi)/180.
        self.def_ang = (def_vel[1] * math.pi)/180.
        
        self.def_velx = self.def_dash_speed * math.cos(self.def_dash_dir)
        self.def_vely = self.def_dash_speed * math.sin(self.def_dash_dir)

    def player_facing(self, angle_between_players):
        """
        Player's relative heading to the defender.
        """
        def_facing = math.degrees(self.defender_pos[2])

        if def_facing < 0:
            def_facing = 180 - abs(def_facing)
            
        elif def_facing >= 0:
            def_facing = 180 + def_facing
            
        if angle_between_players < 0:
            angle_between_players = 360 - abs(angle_between_players)
            
        angle_diff = angle_between_players - def_facing
        return angle_diff

    def train_loop(self):
        attacker_start_pos = np.linspace(start=-3.0, stop=3.0, num=10)
        att_y = np.random.uniform(low=-2.5, high=2.5)
        for episodes in range(total_episodes):
            t_axis = []
            count = 0
            ######### ATTACKER INIT #########
            att_x = -2.0
            if episodes % 2 == 0 and episodes > 0:
                attacker_start_seed = np.random.randint(low=0, high=10)
                att_y = attacker_start_pos[attacker_start_seed]

            self.attacker_pos = np.array([att_x, att_y, 0.0])   # [x, y, theta]
            self.attacker_ang = 0.0
            self.ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])
            self.prev_ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])
            normal_pre_state = np.concatenate((self.attacker_pos, self.att_extra_states, self.ball_pos))
            normal_pre_state = normal_pre_state / np.linalg.norm(normal_pre_state)

            ######### DEFENDER INIT #########
            def_y = 0.0
            def_init_facing = 0.0
            self.defender_pos = np.array([2.0, def_y, def_init_facing])
            init_dist_between_players = math.sqrt((self.attacker_pos[0]-2.0)**2 + (self.attacker_pos[1])**2)
            init_player_facing = math.atan2(self.attacker_pos[1], 4.0)
            self.defender_prev_state = np.array([2.0, def_y, def_init_facing, init_dist_between_players, init_player_facing, 
                                                 init_player_facing, 0.2, 0.0])

            # self.defender_prev_state /= np.linalg.norm(self.defender_prev_state)
            # normal_pre_state = self.defender_prev_state
            ep_reward_list = []

            while True:
                count += 1
                ###### ATTACKER UPDATE #######
                attacker_state = np.concatenate((self.attacker_pos, self.att_extra_states, self.ball_pos))   # TODO: change attacker's input
                attacker_input = tf.expand_dims(tf.convert_to_tensor(attacker_state), 0)
                attacker_action = self.attacker_actor.policy(attacker_input, noise_object=ou_noise)

                self.calc_ball_pos()
                if self.player_to_ball_dist() <= 0.1:
                    self.attacker_state = State.BALL_KICKABLE

                if self.attacker_state == State.BALL_KICKABLE:
                    kick_dir = attacker_action[1]
                    kick_pow = attacker_action[0]
                    self.kick_update((kick_pow, kick_dir))

                    self.attacker_state = State.IDLE

                self.follow_ball(self.ball_pos)

                ball_to_player = self.player_to_ball_dist()
                if (self.player_state == State.DASHING and
                   not self.ball_state == State.KICKING and
                   ball_to_player <= 0.1):
                    rb_facing = (self.attacker_ang * np.pi)/180.
                    self.ball_pos[0] = self.attacker_pos[0] + ball_to_player*math.cos(rb_facing)
                    self.ball_pos[1] = self.attacker_pos[1] + ball_to_player*math.sin(rb_facing)
                    
                elif ball_to_player > 0.1:
                    self.attacker_pos[0] += self.TIME_STEP * self.velx
                    self.attacker_pos[1] += self.TIME_STEP * self.vely

                self.prev_ball_pos = self.ball_pos
                self.player_state = State.STOPPED
                self.ball_state = State.STOPPED
                
                ######## REWARDS #########
                self.attacker_pos[2] = math.radians(self.attacker_ang)   # attacker_ang is in degrees
                rewards, done, self.att_extra_states = self.env.step(self.ball_pos, self.attacker_pos, self.defender_pos)

                # convert defender heading to radians
                normal_state = attacker_state
                normal_state[4] = math.radians(normal_state[4])   # convert player_facing to radians
                normal_state[5] = math.radians(normal_state[5])   # convert angle_between to radians

                # 3-step TD Target
                self.attacker_actor.buffer.record((normal_pre_state, attacker_action, rewards/100, normal_state))
                if (count % 3 == 0 and count > 3) or done:
                    self.attacker_actor.buffer.learn(self.attacker_actor.target_actor, self.attacker_actor.target_critic,
                                self.attacker_actor.actor_model, self.attacker_actor.critic_model,
                                self.attacker_actor.critic_optimizer, self.attacker_actor.actor_optimizer, gamma=gamma)
                    self.attacker_actor.update_actor_target(tau)
                    self.attacker_actor.update_critic_target(tau)

                normal_pre_state = normal_state
                ep_reward_list.append(rewards)

                ###### DEFENDER UPDATE #######
                dist_to_ball = math.sqrt((self.defender_pos[0]-self.ball_pos[0])**2 + (self.defender_pos[1]-self.ball_pos[1])**2)
                angle = math.degrees(math.atan2(self.attacker_pos[1] - self.defender_pos[1], self.attacker_pos[0] - self.defender_pos[0]))
                player_facing = self.player_facing(angle)

                if 0 < self.defender_pos[2] < 3.14:
                    pass
                else:
                    self.defender_pos[2] = math.radians(180 + abs(math.degrees(self.defender_pos[2])))

                defender_input = np.concatenate((self.defender_pos, np.array([dist_to_ball, player_facing, angle, self.ball_pos[0], self.ball_pos[1]])))
                def_state = tf.expand_dims(tf.convert_to_tensor(defender_input), 0)

                defender_action = self.defender_actor.predict(def_state, verbose=0)
                outputs = defender_action[0] * 100.0
                outputs[0] = tf.clip_by_value(outputs[0], 50., 100.)   # dash power
                outputs[1] = tf.clip_by_value(outputs[1], -100., 100.) # dash direction
                dash_speed = -outputs[0]
                dash_dir = outputs[1]

                self.defender_dash((dash_speed, dash_dir)) 
                self.defender_pos[0] += self.TIME_STEP * self.def_velx
                self.defender_pos[1] += self.TIME_STEP * self.def_vely
                self.defender_pos[2] = self.def_ang

                # # DEBUG: collect attacker defender position
                # self.x_pos.append(self.defender_pos[0])
                # self.y_pos.append(self.defender_pos[1])
                # self.att_xpos.append(self.attacker_pos[0])
                # self.att_ypos.append(self.attacker_pos[1])
                # self.theta_list.append(self.defender_pos[2])
                # t_axis.append(count)

                if done:
                    break

            avg_reward = np.mean(ep_reward_list[-5:])
            print("Episode * {} * Avg Reward is ==> {}".format(episodes, avg_reward))
            self.rewards_list.append(avg_reward)
            self.episodes_axis.append(episodes)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('rewards', avg_reward, step=episodes)
                tf.summary.scalar('actor_score', self.attacker_actor.buffer.actor_loss, step=episodes)
                tf.summary.scalar('critic_loss', self.attacker_actor.buffer.critic_loss, step=episodes)
                tf.summary.scalar('actor_gradient_norm', self.attacker_actor.buffer.actor_grad, step=episodes)
                tf.summary.scalar('critic_gradient_norm', self.attacker_actor.buffer.critic_grad, step=episodes)

            # # DEBUG: plot defender position
            # plt.plot(self.x_pos, self.y_pos, '*', label="defender pos")
            # plt.plot(self.att_xpos, self.att_ypos, '*r', label="attacker pos")
            # plt.legend()
            # plt.show()

            # self.x_pos = []
            # self.y_pos = []
            # self.att_xpos = []
            # self.att_ypos = []
            # self.theta_list = []

            if episodes % 1000 == 0 and episodes > 0:
                self.attacker_actor.actor_model.save_weights(save_path + "/trained_model/against_defender/att_v2_actor_checkpt.h5")
                self.attacker_actor.critic_model.save_weights(save_path + "/trained_model/against_defender/att_v2_critic_checkpt.h5")
                self.attacker_actor.target_actor.save_weights(save_path + "/trained_model/against_defender/att_v2_target_actor_checkpt.h5")
                self.attacker_actor.target_critic.save_weights(save_path + "/trained_model/against_defender/att_v2_target_critic_checkpt.h5")

        self.attacker_actor.actor_model.save_weights(save_path + "/trained_model/against_defender/att_v2_actor.h5")
        self.attacker_actor.critic_model.save_weights(save_path + "/trained_model/against_defender/att_v2_critic.h5")
        self.attacker_actor.target_actor.save_weights(save_path + "/trained_model/against_defender/att_v2_target_actor.h5")
        self.attacker_actor.target_critic.save_weights(save_path + "/trained_model/against_defender/att_v2_target_critic.h5")


if __name__ == "__main__":
    train_loop = Train()
    train_loop.train_loop()