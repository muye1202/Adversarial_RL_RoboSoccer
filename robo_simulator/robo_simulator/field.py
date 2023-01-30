import math
import rclpy
import time as t
import yaml
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from turtlesim.msg import Pose
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_path
from enum import Enum, auto


class field(Node):
    """Publishing markers representing the field"""
    def __init__(self):
        super().__init__("field")
        
        self.pub_marker = self.create_publisher(MarkerArray, "~/visualization_marker_array", 1)
        self.kick_sub = self.create_subscription(Point, "~/kick", self.kick_update, 10)
        self.pub_ball_marker = self.create_publisher(Marker, "~/visualization_marker", 1)
        
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # generate marker for arena
        self.marker_arr = MarkerArray()
        self.marker_generate()
        self.broadcaster = TransformBroadcaster(self)
        
        # publish marker for ball
        ## the ball will move shorter distances 
        ## as time progresses and then stops
        self.kick_power = 40.
        self.kick_dir = 0.
        self.ball_posx = 0.1
        self.ball_posy = 0.
        self.last_vel = 0.
        
        self.MAX_KICK_VEL = 10
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.02

        # marker for soccer ball
        self.ball = Marker()
        self.ball_marker()
        
    def kick_update(self, point: Point):
        """kick: [pow, dir]"""
        self.kick_power = point.x
        self.kick_dir = point.y
        
        # when receiving new effective kick
        # the ball will move the farthest this time
        self.last_vel = (self.kick_power/self.MAX_STRENGTH)*self.MAX_KICK_VEL

    def calc_ball_pos(self, last_vel):
        """Calculate ball position at each time step"""
        # update the ball position
        new_x = self.last_vel*self.timer_period*math.cos(self.kick_dir)
        new_y = self.last_vel*self.timer_period*math.sin(self.kick_dir)
        
        self.ball_posx += new_x
        self.ball_posy += new_y

        # the ball moves farthest at first kick
        # then decays to zero.
        # calc new moving dist from last moving dist.
        new_vel = last_vel*self.BALL_DECAY
        self.last_vel = new_vel
        
    def marker_generate(self):
        """Create markers array for arena."""
        # 1st long side of the arena
        arena_size = 11
        x_pos = -0.5*arena_size + 1.5
        y_pos = -0.5*arena_size + 1.5
        z_pos = 0.
        for i in range(arena_size):
            marker_shape = Marker()
            marker_shape.header.frame_id = "nusim/world"
            marker_shape.ns = "long_side_0" + str(i)
            marker_shape.id = 0
            marker_shape.action = 0
            marker_shape.type = 1
            marker_shape.color.r = 0.
            marker_shape.color.g = 1.
            marker_shape.color.b = 1.
            marker_shape.color.a = 1.0
            marker_shape.scale.x = 0.5
            marker_shape.scale.y = 0.5
            marker_shape.scale.z = 0.5
            marker_shape.frame_locked = False

            marker_shape.pose.position.x = x_pos
            marker_shape.pose.position.y = y_pos
            marker_shape.pose.position.z = z_pos
            x_pos += 1

            self.marker_arr.markers.append(marker_shape)

        x_pos = -0.5*arena_size + 1.5
        y_pos = 0.5*arena_size + 1.5
        z_pos = 0.
        # 2nd long side of the arena
        for i in range(arena_size):
            marker_shape = Marker()
            marker_shape.header.frame_id = "nusim/world"
            marker_shape.ns = "long_side_1" + str(i)
            marker_shape.id = 1
            marker_shape.action = 0
            marker_shape.type = 1
            marker_shape.color.r = 0.
            marker_shape.color.g = 1.
            marker_shape.color.b = 1.
            marker_shape.color.a = 1.0
            marker_shape.scale.x = 0.5
            marker_shape.scale.y = 0.5
            marker_shape.scale.z = 0.5
            marker_shape.frame_locked = False

            marker_shape.pose.position.x = x_pos
            marker_shape.pose.position.y = y_pos
            marker_shape.pose.position.z = z_pos
            x_pos += 1

            self.marker_arr.markers.append(marker_shape)

        # 1st y side of the arena
        x_pos = -0.5*arena_size + 1.5
        y_pos = -0.5*arena_size + 1.5
        z_pos = 0.
        for i in range(arena_size):
            marker_shape = Marker()
            marker_shape.header.frame_id = "nusim/world"
            marker_shape.ns = "y_side_0" + str(i)
            marker_shape.id = 2
            marker_shape.action = 0
            marker_shape.type = 1
            marker_shape.color.r = 0.
            marker_shape.color.g = 1.
            marker_shape.color.b = 1.
            marker_shape.color.a = 1.0
            marker_shape.scale.x = 0.5
            marker_shape.scale.y = 0.5
            marker_shape.scale.z = 0.5
            marker_shape.frame_locked = False

            marker_shape.pose.position.x = x_pos
            marker_shape.pose.position.y = y_pos
            marker_shape.pose.position.z = z_pos
            y_pos += 1

            self.marker_arr.markers.append(marker_shape)

        x_pos = 0.5*arena_size + 1.5
        y_pos = -0.5*arena_size + 1.5
        z_pos = 0.
        # 2nd y side of arena
        for i in range(arena_size+1):
            marker_shape = Marker()
            marker_shape.header.frame_id = "nusim/world"
            marker_shape.ns = "y_side_1" + str(i)
            marker_shape.id = 3
            marker_shape.action = 0
            marker_shape.type = 1
            marker_shape.color.r = 0.
            marker_shape.color.g = 1.
            marker_shape.color.b = 1.
            marker_shape.color.a = 1.0
            marker_shape.scale.x = 0.5
            marker_shape.scale.y = 0.5
            marker_shape.scale.z = 0.5
            marker_shape.frame_locked = False

            marker_shape.pose.position.x = x_pos
            marker_shape.pose.position.y = y_pos
            marker_shape.pose.position.z = z_pos
            y_pos += 1

            self.marker_arr.markers.append(marker_shape)
    
    def ball_marker(self):
        self.ball.header.frame_id = "nusim/world"
        self.ball.ns = "soccer"
        self.ball.id = 0
        self.ball.type = 2
        self.ball.action = 0
        self.ball.color.a = 1.
        self.ball.color.r = 0.
        self.ball.color.g = 1.
        self.ball.color.b = 0.
        self.ball.scale.x = 0.12
        self.ball.scale.y = 0.12
        self.ball.scale.z = 0.12
        self.ball.frame_locked = False
        self.ball.pose.position.x = self.ball_posx
        self.ball.pose.position.y = self.ball_posy
    
    def timer_callback(self):
        # publish arena marker
        self.pub_marker.publish(self.marker_arr)
        
        # update ball position at each cycle
        # get kick pow and dir from subsriber
        self.calc_ball_pos(self.last_vel)
        self.get_logger().info("ball velocity: " + str(self.last_vel))
        self.ball_marker()
        self.pub_ball_marker.publish(self.ball)


def main(args=None):
    rclpy.init(args=args)
    arena_pub = field()
    rclpy.spin(arena_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
