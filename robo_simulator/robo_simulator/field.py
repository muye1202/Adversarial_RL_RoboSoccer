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
        
        self.pub_marker = self.create_publisher(MarkerArray, "/visualization_marker_array", 1)
        # self.pub_brick_marker = self.create_publisher(Marker, "/visualization_marker", 1)
        
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # generate marker for arena
        self.marker_arr = MarkerArray()
        self.marker_generate()
        self.broadcaster = TransformBroadcaster(self)
        
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
            
    def timer_callback(self):
        # publish arena marker
        self.pub_marker.publish(self.marker_arr)


def main(args=None):
    rclpy.init(args=args)
    arena_pub = field()
    rclpy.spin(arena_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    

# publish brick frame
# time = self.get_clock().now().to_msg()
# odom_brick = TransformStamped()
# odom_brick.header.frame_id = "world"
# odom_brick.header.stamp = time
# odom_brick.child_frame_id = "brick"

# # if the brick is falling, change its height
# if self.brick_state == State.FALLING:
#     # the brick is falling with acceleration
#     brick_time = t.time()
#     falling_dist = 0.5 * self.g * (brick_time - self.start_time)**2
#     self.curr_height = self.og_z - falling_dist
#     odom_brick.transform.translation.z = self.curr_height
#     odom_brick.transform.translation.x = self.og_x
#     odom_brick.transform.translation.y = self.og_y
#     self.broadcaster.sendTransform(odom_brick)

# # distance between brick and platform
# d = math.sqrt((self.curr_height - self.platform_h)**2 +
#               (self.og_x - self.turtle_pos.x)**2 + (self.og_y - self.turtle_pos.y)**2)
# # determine whether brick touches platform OR ground
# if d <= 0.3 and not self.start_tilting:
#     self.brick_state = State.TOUCH

#     state_change = Empty()
#     self.state_pub.publish(state_change)

# elif self.curr_height <= 0.2 and self.brick_state == State.FALLING:

#     self.brick_state = State.ON_GROUND

# # determine if platform starts tilting
# # only enters once
# if (self.tilt_platform != 0) and self.brick_state == State.TOUCH:

#     if not self.start_tilting:

#         self.start_tilt = t.time()
#         self.brick_state = State.TILT
#         self.tilt_start_y = self.brick_y
#         self.tilt_start_z = self.temp_h
#         self.start_tilting = True

# # generate brick marker with its position
# # calculate brick pos with platform angle
# if self.brick_state == State.TILT:

#     brick_time = t.time()
#     # re-orient brick
#     self.platform_joint_state = self.tilt_platform
#     self.brick_orientation = self.platform_joint_state

#     if self.platform_joint_state < 0:
#         direction = 1
#     else:
#         direction = -1

#     y_s = direction*0.5*(self.g * math.sin(abs(self.platform_joint_state)) * (brick_time -
#                          self.start_tilt)**2) * math.cos(self.platform_joint_state)
#     self.brick_y = self.tilt_start_y + y_s
#     self.y_slide = abs(y_s)

#     z_s = 0.5*(self.g * math.sin(abs(self.platform_joint_state)) * (brick_time -
#                self.start_tilt)**2) * math.sin(abs(self.platform_joint_state))
#     self.curr_height = self.tilt_start_z - z_s
#     self.z_slide = z_s

#     slide_dist = math.sqrt(self.z_slide**2 + self.y_slide**2)

#     if slide_dist > self.platform_r + 0.5*self.brick_size_y:
#         self.brick_state = State.CLEAR

# self.brick_marker = Marker()
# brick_x = self.brick_x
# brick_y = self.brick_y

# self.brick_marker.header.frame_id = "/world"
# self.brick_marker.ns = "brick"
# self.brick_marker.id = 0
# self.brick_marker.type = 1
# self.brick_marker.action = 0
# self.brick_marker.color.a = 1.
# self.brick_marker.color.r = 0.
# self.brick_marker.color.g = 1.
# self.brick_marker.color.b = 0.
# self.brick_marker.scale.x = self.brick_size_x
# self.brick_marker.scale.y = self.brick_size_y
# self.brick_marker.scale.z = self.brick_size_z
# self.brick_marker.frame_locked = False
# self.brick_marker.pose.position.x = brick_x
# self.brick_marker.pose.position.y = brick_y

# if self.brick_state == State.STOPPED:
#     self.brick_marker.pose.position.z = self.og_z
#     self.brick_marker.pose.position.x = self.og_x
#     self.brick_marker.pose.position.y = self.og_y
# elif self.brick_state == State.ON_GROUND:
#     self.brick_marker.pose.position.z = 0.0
#     self.brick_marker.pose.position.x = self.og_x
#     self.brick_marker.pose.position.y = self.og_y
# elif self.brick_state == State.FALLING:
#     self.brick_marker.pose.position.z = self.curr_height
# elif self.brick_state == State.TILT:
#     self.brick_marker.pose.position.z = self.curr_height
#     self.brick_marker.pose.orientation.x = self.brick_orientation

# elif self.brick_state == State.TOUCH:
#     self.brick_marker.pose.position.z = self.platform_h + self.brick_marker.scale.z/2
#     self.brick_marker.pose.position.x = self.turtle_pos.x
#     self.brick_marker.pose.position.y = self.turtle_pos.y

#     self.brick_x = self.turtle_pos.x
#     self.brick_y = self.turtle_pos.y

#     self.temp_h = self.platform_h + self.brick_marker.scale.z/2
#     odom_brick.transform.translation.z = self.temp_h
#     odom_brick.transform.translation.x = self.turtle_pos.x
#     odom_brick.transform.translation.y = self.turtle_pos.y
#     self.broadcaster.sendTransform(odom_brick)