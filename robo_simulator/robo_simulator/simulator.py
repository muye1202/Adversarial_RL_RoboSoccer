import rclpy
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D, Quaternion, Point
from ament_index_python.packages import get_package_share_path
from enum import Enum, auto


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
        
        # send kick cmd
        self.kick_pub = self.create_publisher(Point, "field/kick", 10)
        
        # send robot velocity cmd
        self.vel_pub = self.create_publisher(Pose2D, "field/player_vel", 10)
        
        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "field/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "field/robot_pos", self.robot_callback, 10)
        
        # field params
        self.ball_pos = Point()
        self.robot_pos = Pose2D()
        self.robo_state = State.IDLE
    
    def ball_callback(self, ball_pos: Point):
        self.ball_pos = ball_pos
        
    def robot_callback(self, robo_pos: Pose2D):
        self.robot_pos = robo_pos
        self.get_logger().info("robot direction: " + str(robo_pos.theta))
        
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
        robot_facing = self.robot_pos.theta
        
        self.get_logger().info("robot direction: " + str(robot_facing) + " " + str(ball_to_robot))
        
        if robot_facing <= 0 and 0 <= ball_to_robot <= 90:
            if -(180-ball_to_robot) < robot_facing:
                return abs(robot_facing) + ball_to_robot
            else:
                return -(360 - abs(robot_facing) - ball_to_robot)
            
        elif robot_facing <= 0 and 90 < ball_to_robot <= 180:
            if ball_to_robot - robot_facing < 180:
                return ball_to_robot - robot_facing
            else:
                return -(360-ball_to_robot - robot_facing)
            
        elif 180 >= robot_facing >= 0 and 0 <= ball_to_robot <= 180:
            if robot_facing > ball_to_robot:
                return -(robot_facing-ball_to_robot)
            else:
                return ball_to_robot - robot_facing
        
        elif 180 >= robot_facing >= 0 and ball_to_robot <= 0:
            if abs(ball_to_robot) + robot_facing < 180:
                return -abs(ball_to_robot) - robot_facing
            else:
                return 360 - (abs(ball_to_robot) + robot_facing)
            
        return 0
        
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
        
        # TEST: kick the ball and run to it
        if self.player_to_ball_dist() <= 0.1:
            self.robo_state = State.BALL_KICKABLE

        if self.robo_state == State.BALL_KICKABLE:
            kick_dir = np.random.uniform(low=-55, high=55)
            self.kick(70., kick_dir)
            self.robo_state = State.IDLE
        
        self.follow_ball(self.ball_pos)

def main(args=None):
    rclpy.init(args=args)
    robot_pub = rs_simulator()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()