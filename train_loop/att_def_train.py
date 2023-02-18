import math
import sys
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from enum import Enum, auto
from rl.ddpg_robo_soccer import OUActionNoise, DDPG_robo


"""
Training loop for the RL network.
"""
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

total_episodes = 5000
# Used to update target networks
tau = 0.005
gamma = 0.99

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

kickpow_upper_bound = 70.
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
    arena_range_y = arena_range_x/2
    
    def __init__(self) -> None:
        self.ball_pos = np.array([0.2, 0.0])
        self.attacker_pos = np.array([0.0, 0.0])
        self.defender_pos = np.array([2.0, 0.0, 0.0])
        self.last_defender_pos = np.array([2.0, 0.0, 0.0])
        self.last_attacker_pos = np.array([0.0, 0.0, 0.0])
        self.last_dist_players = 2.0
        self.count_steps = 0
        
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos[0] - self.attacker_pos[0])**2 + 
                         (self.ball_pos[1] - self.attacker_pos[1])**2)
    
    def step(self, ball_pos, attacker_pos, defender_pos):
        done = False
        rewards = 0.0
        self.ball_pos = ball_pos
        self.attacker_pos = attacker_pos
        self.defender_pos = defender_pos

        in_tackle_range = False
        ### DEFENDER REWARDS ###
        # For each step without being scored, reward = +1
        # Taking control over the ball, reward = +10
        # Blocking shooting route, reward = +1

        # stop the defender when it runs pass the attacker
        if not self.is_scored() and self.is_out_of_range():
            rewards = -1.0
            done = True
            
        elif self.defender_pos[0] < 1.0+self.attacker_pos[0]:
            done = True

        if self.dist_between_players() <= 0.8 and self.is_player_facing(role="attacker", del_angle=2.0):
            rewards += 2.0
            done = True
            
            in_tackle_range = True

        elif self.dist_between_players() <= 1.6 and self.is_player_facing(role="attacker", del_angle=5.0):
            rewards += 1.0
            done = True

            if (abs(math.degrees(self.defender_pos[2]) - math.degrees(self.last_defender_dir)) <= 60.0):
                rewards += 0.1
                
            in_tackle_range = True

        # if the player chases the opponent directly
        # it will obtain the max rewards
        if abs(self.attacker_pos[0] - self.defender_pos[0]) <= 3.2 and self.chasing_score() >= 0.97:
            rewards += 2.0
            
        elif abs(self.attacker_pos[0] - self.defender_pos[0]) <= 3.2 and self.chasing_score() >= 0.95:
            rewards += 0.5
        
        if self.count_steps >= 35 and in_tackle_range:
            rewards += 1.0
                            
            done = True
            self.count_steps = 0

        self.last_dist_players = self.dist_between_players()
        self.last_attacker_pos = self.attacker_pos
        self.last_defender_pos = self.defender_pos
        self.count_steps += 1
        self.last_defender_dir = self.defender_pos[2]

        return rewards, done

    def angle_between(self):
        """
        Angle between players.
        """
        
        return math.degrees(math.atan2(self.attacker_pos[1] - self.defender_pos[1],
                                       self.attacker_pos[0] - self.defender_pos[0]))

    def is_player_facing(self, role, del_angle):
        """
        Is player facing the target (compare angles).
        """

        # check if defender can "see" attacker
        angle_between_players = self.angle_between()
        def_facing = math.degrees(self.defender_pos[2])
        attacker_facing = math.degrees(self.attacker_pos[2])

        if role == "attacker":
            if def_facing > 180.0:
                def_facing = -(360 - def_facing)
                
            angle_diff = abs(def_facing - angle_between_players)
            return (angle_diff <= del_angle)
            
        # check if the attacker is running into the defender
        if attacker_facing > 180.0:
            attacker_facing = -(360 - attacker_facing)
                
        angle_diff = abs(attacker_facing - angle_between_players)
        return (angle_diff <= del_angle)

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

        vect_to_curr = np.array([self.defender_pos[0] - self.last_defender_pos[0],
                                 self.defender_pos[1] - self.last_defender_pos[1]])
        dot_product = np.dot(vect_to_curr, vect_to_opponent)

        return dot_product / (np.linalg.norm(vect_to_curr)*np.linalg.norm(vect_to_opponent))


class Train():
    def __init__(self):
        # attacker params
        self.attacker_pos = np.array([0., 0., 0.0])
        self.ball_pos = np.array([0.2, 0.0])
        self.MAX_KICK_VEL = 30
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.4
        self.PLAYER_MAX_SPEED = 1.0
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
        self.attacker_actor = DDPG_robo(0.0,0.0,0.0,0.0,num_states=2,flag="predict")
        self.attacker_actor.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_attacker/attacker_actor.h5")

        # defender params
        self.env = DefenderEnv()
        self.DEFENDER_MAX_SPEED = 1.0
        self.def_velx = 0.0
        self.def_vely = 0.0
        self.defender_pos = np.array([2.0, 0.0, 1.57])
        self.defender_prev_state = np.array([2.0, 0.0, 1.57, 0.2, 0.0])
        self.defender_actor = DDPG_robo(first_low=20., first_high=50., sec_low=-100, sec_high=100, num_states=5, flag="")
        
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
        dash_pow = 80.
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
            self.attacker_pos = np.array([-2.0, 0., 0.0])   # [x, y, theta]
            self.attacker_ang = 0.0
            self.attacker_pos[1] = np.random.uniform(low=-0.5, high=0.5)
            self.ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])
            self.prev_ball_pos = np.array([self.attacker_pos[0]+0.2, self.attacker_pos[1]])
            
            self.defender_pos = np.array([2.0, 0.0, 1.57])
            self.defender_prev_state = np.concatenate((self.defender_pos, self.ball_pos))
            defender_episodic_rewards = 0.0

            while True:
                ###### ATTACKER UPDATE #######
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_ball_pos), 0)
                action = self.attacker_actor.actor_model.predict(tf_prev_state, verbose=0)
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
                defender_input = np.concatenate((self.defender_pos, self.ball_pos))
                defender_input = tf.expand_dims(tf.convert_to_tensor(defender_input), 0)
                defender_action = self.defender_actor.policy(defender_input, noise_object=ou_noise)
                dash_speed = defender_action[0]

                dash_dir = 180.0 + defender_action[1]
                if dash_dir > 180.0:
                    dash_dir = -(360.0 - dash_dir)
                
                self.defender_dash((dash_speed, dash_dir))
                self.defender_pos[0] += self.TIME_STEP * self.def_velx
                self.defender_pos[1] += self.TIME_STEP * self.def_vely
                
                ######## REWARDS #########
                self.attacker_pos[2] = self.attacker_ang
                rewards, done = self.env.step(self.ball_pos, self.attacker_pos, self.defender_pos)
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
                
                #print("the defender pos: " + str(self.defender_pos[0]) + " " + str(self.defender_pos[1]))
            
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(episodes, avg_reward))

            if episodes % 1000 == 0 and episodes > 0:
                self.defender_actor.actor_model.save_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_vs_one/defender_actor_10000.h5")

               
if __name__ == "__main__":
    train_loop = Train()
    train_loop.train_loop()