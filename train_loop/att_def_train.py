import math
import sys
sys.path.insert(0, '/home/muye/rl_soccer')
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from enum import Enum, auto
from rl.ddpg_robo_soccer import OUActionNoise, DDPG_robo


"""
Training loop for the RL network.
"""
std_dev = 0.05
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

total_episodes = 5000
# Used to update target networks
tau = 0.002
gamma = 0.99

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
    
    
class DefenderEnv():
    arena_range_x = 3.5
    arena_range_y = arena_range_x
    
    def __init__(self) -> None:
        self.ball_pos = np.array([0.2, 0.0])
        self.attacker_pos = np.array([0.0, 0.0])
        self.defender_pos = np.array([2.0, 0.0, 0.0])
        self.last_defender_pos = np.array([2.0, 0.0, 0.0])
        self.last_attacker_pos = np.array([0.0, 0.0, 0.0])
        self.last_dist_players = 2.0
        self.count_steps = 0
        self.last_dist_to_start = 5.0
        
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos[0] - self.attacker_pos[0])**2 + 
                         (self.ball_pos[1] - self.attacker_pos[1])**2)
    
    def step(self, ball_pos, attacker_pos, defender_pos):
        done = False
        rewards = 0.0
        self.ball_pos = ball_pos
        self.attacker_pos = attacker_pos
        self.defender_pos = defender_pos

        self.count_steps += 1
        ### DEFENDER REWARDS ###
        # stop the defender when it runs pass the attacker
        if self.is_scored():
            rewards -= 3000.0
            done = True

        # stop the defender when it runs pass the attacker
        if not self.is_scored() and self.is_out_of_range():
            rewards = -3000.0
            done = True

        if self.dist_between_players() <= 0.8:
            rewards += 500.0

        if (self.dist_between_players() <= 2.4 and abs(self.player_facing()) < 10.0):
            rewards += 500.0 * 15.0 / (self.player_facing() + 1)

        elif (self.dist_between_players() <= 2.4 and abs(self.player_facing()) < 15.0):
            rewards += -20*self.player_facing() + 600

        elif (self.dist_between_players() <= 2.4 and abs(self.player_facing()) > 30.0):
            rewards -= 50.0

        # if the player chases the opponent directly
        # it will obtain the max rewards
        if self.count_steps == 5:

            if (self.dist_between_players() <= 2.4 and self.chasing_score() < 5.0):
                if self.last_dist_to_start < self.dist_to_start():
                    rewards += 150.0
                else:
                    rewards += 100.0

            elif (self.dist_between_players() <= 2.4 and self.chasing_score() < 7.0):
                if self.last_dist_to_start < self.dist_to_start():
                    rewards += 70.0
                else:
                    rewards += 40.0

            elif (self.dist_between_players() <= 2.0 and self.chasing_score() > 15.0):
                rewards -= 50.0

            self.count_steps = 0

        if self.ball_to_goal_dist() <= 2.0:
            rewards -= 500.0

        # punish if the player goes over the attacker
        # without being in range    
        if (self.defender_pos[0] < self.attacker_pos[0]+0.1):
            done = True
            if (self.dist_between_players() >= 1.5):
                rewards -= 900.0

        self.last_dist_players = self.dist_between_players()
        self.last_attacker_pos = self.attacker_pos
        self.last_defender_pos = self.defender_pos
        self.last_defender_dir = self.defender_pos[2]
        self.last_dist_to_start = self.dist_to_start()   # distance to the other side of the field
        extra_states = np.array([self.dist_between_players(), self.player_facing(), self.angle_between()])

        return rewards, done, extra_states

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
        def_facing = self.defender_pos[2]

        # if def_facing < 0:
        #     def_facing = 180 - abs(def_facing)

        # elif def_facing >= 0:
        #     def_facing = 180 + def_facing

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
        robot_facing = math.degrees(self.attacker_pos[2])

        return (robot_facing <= left_post_angle and
                robot_facing >= right_post_angle)
        
    def is_out_of_range(self):
        """
        Determine if the defender is out of field.
        
        Note: the episode will end if the defender run pass the attacker.
        """

        return (self.defender_pos[0] <= -self.arena_range_x-0.2 or
                self.defender_pos[0] >= self.arena_range_x+0.2 or
                self.defender_pos[1] <= -self.arena_range_y-0.2 or
                self.defender_pos[1] >= self.arena_range_y+0.2)

    def chasing_score(self):
        """
        Check how much does the player advance in the direction of the opponent.
        
        Calculate the dot product between the vector from the player to the opponent,
        and the vector from its last to current position; then assign rewards proportional
        to the dot product
        """
        vect_to_opponent = np.array([self.attacker_pos[0] - self.defender_pos[0], 
                                     self.attacker_pos[1] - self.defender_pos[1]])
        vect_to_opponent = vect_to_opponent / np.linalg.norm(vect_to_opponent)

        vect_to_curr = np.array([self.defender_pos[0] - self.last_defender_pos[0],
                                 self.defender_pos[1] - self.last_defender_pos[1]])
        vect_to_curr = vect_to_curr / np.linalg.norm(vect_to_curr)
        dot_product = np.dot(vect_to_curr, vect_to_opponent)

        return math.degrees(np.arccos(dot_product))


def attacker_actor():
    """
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(2)))
    model.add(layers.Dense(256, activation="relu", use_bias=True))
    model.add(layers.Dense(256, activation="tanh", use_bias=True))
    model.add(layers.Dense(2, activation="tanh", use_bias=True, kernel_initializer=last_init))

    # # attacker is trained with 3 layered NN
    # inputs = layers.Input(shape=(2))
    # out = layers.Dense(256, activation="relu", use_bias=True)(inputs)
    # out = layers.Dense(256, activation="tanh", use_bias=True)(out)
    # outputs = layers.Dense(num_actions, activation="tanh", use_bias=True, kernel_initializer=last_init)(out)

    # max_num = np.array([first_low, first_high, sec_low, sec_high])
    # outputs = outputs * np.amax(max_num)

    # model = tf.keras.Model(inputs, outputs)
    return model


class Train():
    def __init__(self):
        # attacker params
        self.attacker_pos = np.array([0., 0., 0.0])
        self.ball_pos = np.array([0.2, 0.0])
        self.MAX_KICK_VEL = 30
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.4
        self.PLAYER_MAX_SPEED = 0.8
        self.TIME_STEP = 0.1  # simulator timestep 1.0 second
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
        self.attacker_actor = attacker_actor()
        self.attacker_actor.load_weights("/home/muye/rl_soccer/successful_model/one_attacker/attacker_actor_2000.h5")

        # defender params
        self.env = DefenderEnv()
        self.DEFENDER_MAX_SPEED = 1.0
        self.def_velx = 0.0
        self.def_vely = 0.0
        self.defender_pos = np.array([2.0, 0.0, 1.57])
        self.defender_prev_state = np.array([2.0, 0.0, 1.57, 4.0, 0.0, 0.0, 0.2, 0.0])   # [x, y, theta, dist_between_players, 
                                                                                        #  player_facing, angle_between, ball_x, ball_y]
        self.extra_states = np.array([4.0, 0.0, 0.0])
        self.defender_actor = DDPG_robo(first_low=40., first_high=70., sec_low=-100, sec_high=100, num_states=8, flag="defender")
        
        # # DEBUG:
        # self.x_pos = []
        # self.y_pos = []
        # self.theta_list = []
        # self.att_xpos = []
        # self.att_ypos = []
        
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
    
    def train_loop(self):
        for episodes in range(total_episodes):
            t_axis = []
            count = 0
            ######### ATTACKER INIT #########
            att_x = -2.0
            att_y = np.random.uniform(low=-3.0, high=3.0)
            self.attacker_pos = np.array([att_x, att_y, 0.0])   # [x, y, theta]
            self.attacker_ang = 0.0
            self.ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])
            self.prev_ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])

            ######### DEFENDER INIT #########
            self.defender_pos = np.array([2.0, 0.0, 1.57])
            init_dist_between_players = math.sqrt((self.attacker_pos[0]-2.0)**2 + (self.attacker_pos[1])**2)
            init_player_facing = math.atan2(self.attacker_pos[1], 4.0)
            self.defender_prev_state = np.array([2.0, 0.0, 1.57, init_dist_between_players, init_player_facing, 
                                                 init_player_facing, 0.2, 0.0])
            defender_episodic_rewards = 0.0

            while True:
                count += 1
                ###### ATTACKER UPDATE #######
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_ball_pos), 0)
                action = (self.attacker_actor(tf_prev_state)).numpy()
                outputs = action[0] * kickdir_high
                outputs[0] = tf.clip_by_value(outputs[0], kickpow_lower_bound, kickpow_upper_bound)   # kick power
                outputs[1] = tf.clip_by_value(outputs[1], kickdir_low, kickdir_high) # kick direction


                self.calc_ball_pos()
                if self.player_to_ball_dist() <= 0.1:
                    self.attacker_state = State.BALL_KICKABLE

                if self.attacker_state == State.BALL_KICKABLE:
                    kick_dir = outputs[1]
                    kick_pow = outputs[0]
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

                ###### DEFENDER UPDATE #######
                defender_input = np.concatenate((self.defender_pos, self.extra_states))
                defender_input = np.concatenate((defender_input, self.ball_pos))
                defender_input = tf.expand_dims(tf.convert_to_tensor(defender_input), 0)
                defender_action = self.defender_actor.policy(defender_input, noise_object=ou_noise)
                dash_speed = -defender_action[0]
                dash_dir = defender_action[1]

                self.defender_dash((dash_speed, dash_dir)) 
                self.defender_pos[0] += self.TIME_STEP * self.def_velx
                self.defender_pos[1] += self.TIME_STEP * self.def_vely

                # if dash_dir < 0:
                #     self.defender_pos[2] = 180 + dash_dir
                self.defender_pos[2] = 180.0 + dash_dir

                ######## REWARDS #########
                self.attacker_pos[2] = self.attacker_ang
                rewards, done, self.extra_states = self.env.step(self.ball_pos, self.attacker_pos, self.defender_pos)

                # convert defender heading to radians
                normal_state = defender_input
                normal_pre_state = self.defender_prev_state
                # normalize input
                if np.linalg.norm(defender_input) > 0 and np.linalg.norm(self.defender_prev_state) > 0:
                    normal_state = defender_input / np.linalg.norm(defender_input)
                    normal_pre_state = self.defender_prev_state / np.linalg.norm(self.defender_prev_state)

                self.defender_actor.buffer.record((normal_pre_state, defender_action, rewards, normal_state))
                defender_episodic_rewards += rewards

                self.defender_actor.buffer.learn(self.defender_actor.target_actor, self.defender_actor.target_critic,
                             self.defender_actor.actor_model, self.defender_actor.critic_model,
                             self.defender_actor.critic_optimizer, self.defender_actor.actor_optimizer, gamma=gamma)
                self.defender_actor.update_actor_target(tau)
                self.defender_actor.update_critic_target(tau)
                
                if done:
                    break
                
                self.defender_prev_state = defender_input
                ep_reward_list.append(defender_episodic_rewards)
                
                # # DEBUG: plot defender position
                # self.x_pos.append(self.defender_pos[0])
                # self.y_pos.append(self.defender_pos[1])
                # self.att_xpos.append(self.attacker_pos[0])
                # self.att_ypos.append(self.attacker_pos[1])
                # self.theta_list.append(self.defender_pos[2])
                # t_axis.append(count)

            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(episodes, avg_reward))

            # # DEBUG: plot defender position
            # plt.plot(self.x_pos, self.y_pos, '*', label="defender pos")
            # plt.plot(self.att_xpos, self.att_ypos, '*r', label="attacker pos")
            # plt.legend()
            # plt.show()

            # plt.plot(t_axis, self.theta_list, label="defender heading")
            # plt.legend()
            # plt.show()
            # self.x_pos = []
            # self.y_pos = []
            # self.att_xpos = []
            # self.att_ypos = []
            # self.theta_list = []
        
            if episodes % 1000 == 0 and episodes > 0:
                self.defender_actor.actor_model.save_weights("/home/muye/rl_soccer/trained_model/one_vs_one/defender_actor_5000.h5")

               
if __name__ == "__main__":
    train_loop = Train()
    train_loop.train_loop()