from std_msgs.msg import Empty
import rclpy
import math
import yaml
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D, Vector3, Point, TwistWithCovariance
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_path
from enum import Enum, auto

class rs_simulator(Node):
    """Publish a robot position and ball marker position"""
    
    def __init__(self):
        super().__init__("rs_simulator")
        
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # receive the velocity for the robot
        self.vel_sub = self.create_subscription(Pose2D, '~/player_vel', self.vel_callback, 10)
        
        # publish the location of the robot
        self.broadcaster = TransformBroadcaster(self)
        
        # robo action param
        self.startx = 0.
        self.starty = 0.
        self.posx = 0.
        self.posy = 0.
        self.ang = 0.
        
    def vel_callback(self, pose: Pose2D):
        self.posx = pose.x
        self.posy = pose.y
        self.ang = pose.theta   
    
    def timer_callback(self):
        self.startx += 0.01

        # publish transform for the robot
        time = self.get_clock().now().to_msg()
        world_robot = TransformStamped()
        world_robot.header.stamp = time
        world_robot.header.frame_id = "nusim/world"
        world_robot.child_frame_id = "purple/base_footprint"
        world_robot.transform.translation.x = self.startx
        world_robot.transform.translation.y = self.starty
        self.broadcaster.sendTransform(world_robot)


def main(args=None):
    rclpy.init(args=args)
    robot_pub = rs_simulator()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()