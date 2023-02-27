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
from rl.ddpg_robo_soccer import DDPG_robo


kickpow_upper_bound = 70.
kickpow_lower_bound = 40.
kickdir_low = -90.
kickdir_high = 90.

class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()


class Defender_Evaluate(Node):
    
    def __init__(self):
        super().__init__("defender_eval")
        
        timer_period = 0.01
        self.timer_ = self.create_timer(timer_period, self.timer_callback)
        
        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "one_one/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "one_one/robot_pos", self.robot_callback, 10)
        self.defender_sub = self.create_subscription(Pose2D, "one_one/defender_pos", self.defender_callback, 10)

        # send kick cmd
        self.kick_pub = self.create_publisher(Point, "one_one/kick", 10)
        
        # send robot velocity cmd
        self.vel_pub = self.create_publisher(Pose2D, "one_one/player_vel", 10)
        # send defender dash cmd
        self.defender_vel_pub = self.create_publisher(Pose2D, "one_one/defender_vel", 10)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Pose2D, "one_one/reset_flag", 10)
        
        # field params
        self.ball_pos = Point()
        self.ball_pos.x = 0.2
        self.ball_pos.y = 0.0
        self.robo_state = State.IDLE
        self.robot_pos = Pose2D()
        self.arena_range_x = 3.5
        self.arena_range_y = self.arena_range_x/2
        self.defender_pos = Pose2D()
        self.new_pos = Pose2D()

        # load model
        self.actor_model = DDPG_robo(0., 0., 0., 0., num_states=2, flag="predict")
        self.actor_model.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/one_attacker/attacker_actor.h5")
        self.defender_model = DDPG_robo(0.,0.,0.,0., num_states=8, flag="defender_predict")
        self.defender_model.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/Feb_26_trained_alien.h5", by_name=True)

    def defender_callback(self, def_pos: Pose2D):
        self.defender_pos = def_pos
        
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
        dash_pow = 50.
        self.dash(dash_pow, dash_dir)
        
    def reset_signal(self):
        """
        Send the reset signal to simulator to reset the position
        of the robot and the ball.
        """
        self.new_pos.x = -2.0
        att_y = np.linspace(start=-3.0, stop=3.0, num=10)
        indx = np.random.randint(low=0, high=10)
        self.new_pos.y = att_y[indx]
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
        
    def player_facing(self, angle_between_players):
        """
        Is player facing the target (compare angles).
        """
        def_facing = math.degrees(self.defender_pos.theta)

        if def_facing < 0:
            def_facing = 180 - abs(def_facing)
            
        elif def_facing >= 0:
            def_facing = 180 + def_facing
            
        if angle_between_players < 0:
            angle_between_players = 360 - abs(angle_between_players)
            
        angle_diff = angle_between_players - def_facing
        return angle_diff
        
    def dist_between_players(self):
        """
        Calculate distance between attacker and defender
        """
        
        return math.sqrt((self.robot_pos.x - self.defender_pos.x)**2
                + (self.robot_pos.y - self.defender_pos.y)**2)
        
    def timer_callback(self):
        ###### SENDING THE COMMANDS ######
        if (self.is_dead_ball() or self.is_scored()
            or self.dist_between_players() <= 0.8):
            self.reset_signal()
        
        else:
            ######### SEND ATTACKER COMMANDS #########
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
                
                self.robo_state = State.IDLE
            
            self.follow_ball(self.ball_pos)

            ######### SEND DEFENDER COMMANDS #########
            # feed in the normalized defender input
            dist_to_ball = math.sqrt((self.defender_pos.x-self.ball_pos.x)**2 + (self.defender_pos.y-self.ball_pos.y)**2)
            angle = math.degrees(math.atan2(self.robot_pos.y - self.defender_pos.y, self.robot_pos.x - self.defender_pos.x))
            player_facing = self.player_facing(angle)
            defender_pos = np.array([self.defender_pos.x, self.defender_pos.y, self.defender_pos.theta])
            
            if 0 < defender_pos[2] < 3.14:
                pass
            else:
                defender_pos[2] = math.radians(180 + abs(math.degrees(defender_pos[2])))

            defender_input = np.concatenate((defender_pos, np.array([dist_to_ball, player_facing, angle, self.ball_pos.x, self.ball_pos.y])))
            # defender_input = defender_input / np.linalg.norm(defender_input)
            
            def_state = tf.expand_dims(tf.convert_to_tensor(defender_input), 0)
            defender_action = self.defender_model.actor_model.predict(def_state, verbose=0)
            outputs = defender_action[0] * 100.0
            outputs[0] = tf.clip_by_value(outputs[0], 50., 100.)   # dash power
            outputs[1] = tf.clip_by_value(outputs[1], -100., 100.) # dash direction
            
            defender_dash = Pose2D()
            defender_dash.x = float(outputs[0])
            defender_dash.y = float(outputs[1])
            self.defender_vel_pub.publish(defender_dash)
        
        
def main(args=None):
    rclpy.init(args=args)
    robot_pub = Defender_Evaluate()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()