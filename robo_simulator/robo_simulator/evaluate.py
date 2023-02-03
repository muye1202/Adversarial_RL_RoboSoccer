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


kickpow_upper_bound = 70.
kickpow_lower_bound = 30.
kickdir_low = -90.
kickdir_high = 90.

class State(Enum):
    """Determine the state of the robot player."""

    BALL_KICKABLE = auto()
    IDLE = auto()


class Evaluate(Node):
    
    def __init__(self):
        super().__init__("evaluate_rs")
        
        timer_period = 0.01
        self.timer_ = self.create_timer(timer_period, self.timer_callback)
        
        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "field/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "field/robot_pos", self.robot_callback, 10)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Empty, "field/reset_flag", 10)
        
        # send kick cmd
        self.kick_pub = self.create_publisher(Point, "field/kick", 10)
        
        # send robot velocity cmd
        self.vel_pub = self.create_publisher(Pose2D, "field/player_vel", 10)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Empty, "field/reset_flag", 10)
        
        # field params
        self.ball_pos = Point()
        self.ball_pos.x = 0.2
        self.ball_pos.y = 0.0
        self.robo_state = State.IDLE
        self.robot_pos = Pose2D()
        self.arena_range_x = 3.5
        self.arena_range_y = self.arena_range_x/2
        
        # load model
        self.actor_model = get_actor()
        self.actor_model.load_weights("/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/trained_model/attacker_actor.h5")
        
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
        dash_pow = 80.
        
        #self.get_logger().info("dash direction: " + str(dash_dir))
        self.dash(dash_pow, dash_dir)
        
    def reset_signal(self):
        """
        Send the reset signal to simulator to reset the position
        of the robot and the ball.
        """
        
        self.reset_pub.publish(Empty())
        
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
        
    def timer_callback(self):
        ###### SENDING THE COMMANDS ######
        if self.is_dead_ball() or self.is_scored():
            self.reset_signal()
        
        else:
            ball_pos = np.array([self.ball_pos.x, self.ball_pos.y])
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(ball_pos), 0)
            action = self.actor_model.predict(tf_prev_state)
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
        
        
def main(args=None):
    rclpy.init(args=args)
    robot_pub = Evaluate()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()