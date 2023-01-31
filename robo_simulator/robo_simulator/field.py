import math
import rclpy
import time as py_time
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Point, Pose2D, Quaternion, TransformStamped
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from enum import Enum, auto


class State(Enum):
    """Determine the state of the robot player."""

    DRIBBLING = auto()
    KICKING = auto()
    DASHING = auto()
    STOPPED = auto()


class field(Node):
    """Publishing markers representing the field"""
    def __init__(self):
        super().__init__("field")
        
        self.pub_marker = self.create_publisher(MarkerArray, "~/visualization_marker_array", 1)
        self.kick_sub = self.create_subscription(Point, "~/kick", self.kick_update, 10)
        self.pub_ball_marker = self.create_publisher(Marker, "~/visualization_marker", 1)
        
        # receive the velocity for the robot
        self.vel_sub = self.create_subscription(Pose2D, '~/player_vel', self.vel_callback, 10)
        
        # send the position of the robot and ball
        self.ball_pos_pub = self.create_publisher(Point, "~/ball_pos", 10)
        self.robot_pos_pub = self.create_publisher(Pose2D, "~/robot_pos", 10)
        
        # reset robot and ball position
        self.reset_srv = self.create_service(Empty, '~/reset', self.reset_callback)
        
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
        self.ball_startx = 0.1
        self.ball_starty = 0.
        self.ball_posx = 0.1
        self.ball_posy = 0.
        self.last_vel = 0.
        
        self.MAX_KICK_VEL = 30
        self.MAX_STRENGTH = 100
        self.BALL_DECAY = 0.4

        # marker for soccer ball
        self.ball = Marker()
        self.ball_marker()
        
        # robo action param
        self.startx = 0.
        self.starty = 0.
        self.posx = 0.
        self.posy = 0.
        self.ang = 0.
        self.velx = 0.
        self.vely = 0.
        self.dash_speed = 0.
        self.dash_dir = 0.
        self.PLAYER_MAX_SPEED = 1
        self.quaternion = Quaternion()
        
        self.player_state = State.STOPPED
        self.ball_state = State.STOPPED
        self.refresh_time = py_time.time()
    
    def reset_callback(self, _, resp):
        """Reset robot and ball positions"""
        self.posx = self.startx
        self.posy = self.starty
        self.ball_posx = self.ball_startx
        self.ball_posy = self.ball_starty
        
        return resp
    
    def vel_callback(self, pose: Pose2D):
        """
        Receive robot dash command
        
        Input:
            - pose: [speed, dir, 0]
        """
        self.dash_speed = (pose.x/100)*self.PLAYER_MAX_SPEED
        self.dash_dir = (pose.y * math.pi)/180.
        self.ang = (pose.y * math.pi)/180.
        
        self.velx = self.dash_speed * math.cos(self.dash_dir)
        self.vely = self.dash_speed * math.sin(self.dash_dir)
        
        quat = quaternion_from_euler(0., 0., self.ang, 'ryxz')
        self.quaternion.x = quat[0]
        self.quaternion.y = quat[1]
        self.quaternion.z = quat[2]
        self.quaternion.w = quat[3]
        self.refresh_time = py_time.time()
        
        # update the player as dashing
        self.player_state = State.DASHING
    
    def kick_update(self, point: Point):
        """kick: [pow, dir]"""
        self.kick_power = point.x
        self.kick_dir = (point.y * math.pi)/180.
        
        # when receiving new effective kick
        # the ball will move the farthest this time
        self.last_vel = (self.kick_power/self.MAX_STRENGTH)*self.MAX_KICK_VEL
        
        self.ball_state = State.KICKING

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
    
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_posx - self.posx)**2 + 
                         (self.ball_posy - self.posy)**2)
    
    def marker_generate(self):
        """Create markers array for arena."""
        # 1st long side of the arena
        arena_size = 5.5
        z_pos = 0.

        marker_shape = Marker()
        marker_shape.header.frame_id = "purple/base_footprint"
        marker_shape.ns = "long_side_0"
        marker_shape.id = 0
        marker_shape.action = 0
        marker_shape.type = 1
        marker_shape.color.r = 0.
        marker_shape.color.g = 1.
        marker_shape.color.b = 1.
        marker_shape.color.a = 1.0
        marker_shape.scale.x = 2*arena_size
        marker_shape.scale.y = 0.2
        marker_shape.scale.z = 0.5
        marker_shape.frame_locked = False

        marker_shape.pose.position.x = 0.
        marker_shape.pose.position.y = -0.5*arena_size
        marker_shape.pose.position.z = z_pos

        self.marker_arr.markers.append(marker_shape)

        marker_shape = Marker()
        marker_shape.header.frame_id = "purple/base_footprint"
        marker_shape.ns = "long_side_1"
        marker_shape.id = 1
        marker_shape.action = 0
        marker_shape.type = 1
        marker_shape.color.r = 0.
        marker_shape.color.g = 1.
        marker_shape.color.b = 1.
        marker_shape.color.a = 1.0
        marker_shape.scale.x = 2*arena_size
        marker_shape.scale.y = 0.2
        marker_shape.scale.z = 0.5
        marker_shape.frame_locked = False

        marker_shape.pose.position.x = 0.
        marker_shape.pose.position.y = 0.5*arena_size
        marker_shape.pose.position.z = z_pos

        self.marker_arr.markers.append(marker_shape)

        # 1st y side of the arena
        marker_shape = Marker()
        marker_shape.header.frame_id = "purple/base_footprint"
        marker_shape.ns = "y_side_0"
        marker_shape.id = 2
        marker_shape.action = 0
        marker_shape.type = 1
        marker_shape.color.r = 0.
        marker_shape.color.g = 1.
        marker_shape.color.b = 1.
        marker_shape.color.a = 1.0
        marker_shape.scale.x = 0.2
        marker_shape.scale.y = arena_size
        marker_shape.scale.z = 0.5
        marker_shape.frame_locked = False

        marker_shape.pose.position.x = -arena_size
        marker_shape.pose.position.y = 0.
        marker_shape.pose.position.z = z_pos

        self.marker_arr.markers.append(marker_shape)

        # goal side marker up
        marker_shape = Marker()
        marker_shape.header.frame_id = "purple/base_footprint"
        marker_shape.ns = "goal_up"
        marker_shape.id = 3
        marker_shape.action = 0
        marker_shape.type = 1
        marker_shape.color.r = 0.
        marker_shape.color.g = 1.
        marker_shape.color.b = 1.
        marker_shape.color.a = 1.0
        marker_shape.scale.x = 0.2
        marker_shape.scale.y = 0.4*arena_size
        marker_shape.scale.z = 0.5
        marker_shape.frame_locked = False

        marker_shape.pose.position.x = arena_size
        marker_shape.pose.position.y = 0.3*arena_size
        marker_shape.pose.position.z = z_pos

        self.marker_arr.markers.append(marker_shape)
        
        # goal side marker down
        marker_shape = Marker()
        marker_shape.header.frame_id = "purple/base_footprint"
        marker_shape.ns = "goal_down"
        marker_shape.id = 3
        marker_shape.action = 0
        marker_shape.type = 1
        marker_shape.color.r = 0.
        marker_shape.color.g = 1.
        marker_shape.color.b = 1.
        marker_shape.color.a = 1.0
        marker_shape.scale.x = 0.2
        marker_shape.scale.y = 0.4*arena_size
        marker_shape.scale.z = 0.5
        marker_shape.frame_locked = False

        marker_shape.pose.position.x = arena_size
        marker_shape.pose.position.y = -0.3*arena_size
        marker_shape.pose.position.z = z_pos

        self.marker_arr.markers.append(marker_shape)
    
    def ball_marker(self):
        self.ball.header.frame_id = "purple/base_footprint"
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
        # calculate time elapsed from last receiving dash cmd
        curr_time = py_time.time()
        robot_travel_time = curr_time - self.refresh_time
        # publish arena marker
        self.pub_marker.publish(self.marker_arr)

        # calculate current robot position
        self.posx += robot_travel_time * self.velx
        self.posy += robot_travel_time * self.vely

        # update ball position at each cycle
        # get kick pow and dir from subsriber
        self.calc_ball_pos(self.last_vel)

        # if the player does not kick, the ball
        # should move with the player
        ball_to_robo = self.player_to_ball_dist()
        if (self.player_state == State.DASHING and 
           not self.ball_state == State.KICKING and
           ball_to_robo <= 0.1):
            # set ball position
            self.ball_posx = self.posx + ball_to_robo*math.cos(self.ang)
            self.ball_posy = self.posy + ball_to_robo*math.sin(self.ang)

        # publish the ball marker
        self.ball_marker()
        self.pub_ball_marker.publish(self.ball)
        
        # publish transform for the robot
        time = self.get_clock().now().to_msg()
        world_robot = TransformStamped()
        world_robot.header.stamp = time
        world_robot.header.frame_id = "purple/base_footprint"
        world_robot.child_frame_id = "purple/base_link"
        world_robot.transform.translation.x = self.posx
        world_robot.transform.translation.y = self.posy
        world_robot.transform.rotation.x = self.quaternion.x
        world_robot.transform.rotation.y = self.quaternion.y
        world_robot.transform.rotation.z = self.quaternion.z
        world_robot.transform.rotation.w = self.quaternion.w
        self.broadcaster.sendTransform(world_robot)
        
        # publish the position of the robot and ball
        r_pos = Pose2D()
        r_pos.x = self.posx
        r_pos.y = self.posy
        r_pos.theta = self.ang
        self.robot_pos_pub.publish(r_pos)
        
        ball_pos = Point()
        ball_pos.x = self.ball_posx
        ball_pos.y = self.ball_posy
        self.ball_pos_pub.publish(ball_pos)
        
        self.player_state = State.STOPPED
        self.ball_state = State.STOPPED


def main(args=None):
    rclpy.init(args=args)
    arena_pub = field()
    rclpy.spin(arena_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
