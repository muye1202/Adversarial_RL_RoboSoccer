import rclpy
from rclpy.node import Node
import numpy as np
import math
from gym.spaces import Box
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Empty
from rl_interfaces.msg import Info


class Defender(Node):
    
    arena_range_x = 3.5
    arena_range_y = arena_range_x/2
    robot_pos = Pose2D()   # this is in radians
    ball_pos = Point()
    defender_pos = Pose2D()   # this is in radians

    def __init__(self):
        super().__init__("defender_env") 
        # state dict: {position of the player, pos of the ball}
        self.state = np.array([0., 0.])
        
        # last ball position
        self.last_ball_dist = self.arena_range_x
        self.last_ball_dist_y = 0.

        #######################################
        timer_period = 0.005
        self.timer_ = self.create_timer(timer_period, self.timer_callback)

        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "one_one/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "one_one/robot_pos", self.robot_callback, 10)
        self.last_dir = 0.
        self.defender_sub = self.create_subscription(Pose2D, "one_one/defender_pos", self.defender_pos_callback, 10)
        
        # publish step function update
        self.defender_pub = self.create_publisher(Info, "defend/defender_info", 1)

        # see if the step function should be called to updtate rewards
        self.def_state_sub = self.create_subscription(Empty, "~/def_update", self.def_update_callback, 10)
        self.def_update = False
        
        # count how many steps have been taken
        self.count_steps = 0.
        
        # last dist between players
        self.last_dist_players = 0.0
        self.last_x_dist = 2.0   # last x dist between players
        self.last_y_dist = 0.0   # last y dist between players
    
    def timer_callback(self):
        if self.def_update:
            self.step()
            
            self.def_update = False
    
    def defender_pos_callback(self, defender_pos: Pose2D):
        self.defender_pos = defender_pos
    
    def def_update_callback(self, _):
        self.def_update = True
    
    def ball_callback(self, ball_pos: Point):
        self.ball_pos = ball_pos
        
    def robot_callback(self, robo_pos: Pose2D):
        self.robot_pos = robo_pos
        
    def player_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos.x - self.robot_pos.x)**2 + 
                         (self.ball_pos.y - self.robot_pos.y)**2)
    
    def step(self):
        done = False
        rewards = 0.0

        ### DEFENDER REWARDS ###
        # For each step without being scored, reward = +1
        # Taking control over the ball, reward = +10
        # Blocking shooting route, reward = +1

        # check whether the robot moves towards
        # the goal in the past episode:
        if self.is_scored():
            rewards -= 10.0
            done = True
            
        if not self.is_scored() and self.is_out_of_range():
            rewards -= 0.1
            done = True

        if self.dist_between_players() <= 0.3 and self.is_player_facing(role="attacker"):
            rewards += 10.0
            done = True

        if (self.dist_between_players() < self.last_dist_players
            and self.is_player_facing(role="attacker")):
            rewards += 0.5

        elif self.is_player_facing(role="attacker"):
            rewards += 0.05

        # if self.chasing_attackers():
        #     rewards += 0.2
        #     self.last_x_dist = self.robot_pos.x - self.defender_pos.x
        #     self.last_y_dist = self.robot_pos.y - self.defender_pos.y

        step_info = Info()
        step_info.states = [self.defender_pos.x, self.defender_pos.y]
        step_info.rewards = rewards
        step_info.done = done
        self.last_dist_players = self.dist_between_players()

        self.defender_pub.publish(step_info)

    def is_player_facing(self, role):
        """
        Is player facing another one
        """

        # check if defender can "see" attacker
        angle_between_players = math.degrees(math.atan2(self.robot_pos.y - self.defender_pos.y,
                                                            self.robot_pos.x - self.defender_pos.x))
        def_facing = math.degrees(self.defender_pos.theta)
        attacker_facing = math.degrees(self.robot_pos.theta)

        if role == "attacker":
            if def_facing > 180.0:
                def_facing = -(360 - def_facing)
                
            angle_diff = abs(def_facing - angle_between_players)
            return (angle_diff <= 10.0)
            
        # check if the attacker is running into the defender
        if attacker_facing > 180.0:
            attacker_facing = -(360 - attacker_facing)
                
        angle_diff = abs(attacker_facing - angle_between_players)
        return (angle_diff <= 10.0)

    def dist_between_players(self):
        """
        Calculate distance between attacker and defender
        """
        
        return math.sqrt((self.robot_pos.x - self.defender_pos.x)**2
                + (self.robot_pos.y - self.defender_pos.y)**2)
        
    def chasing_attackers(self):
        """
        Check whether both x and y distance is decreasing 
        between attackers and defenders.
        """
        curr_x = self.robot_pos.x - self.defender_pos.x
        curr_y = self.robot_pos.y - self.defender_pos.y
        
        return (curr_x < self.last_x_dist and curr_y < self.last_y_dist)

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

    def ball_to_goal_dist(self):
        """
        Calculate the distance between the ball and the goal
        """

        return (self.arena_range_x - self.ball_pos.x)**2 + (self.ball_pos.y)**2

    def is_player_facing_goal(self):
        """
        Determine if the robot is facing the goal
        """

        # calculate the angle between goal's left post
        # and the robot
        left_post_angle = math.degrees(math.atan2(0.4*2*self.arena_range_y-self.robot_pos.y,
                                                  self.arena_range_x-self.robot_pos.x))

        right_post_angle = math.degrees(math.atan2(self.robot_pos.y+0.4*2*self.arena_range_y,
                                                  self.arena_range_x-self.robot_pos.x))
        robot_facing = math.degrees(self.robot_pos.theta)

        return (robot_facing <= left_post_angle and
                robot_facing >= right_post_angle)
        
    def is_out_of_range(self):
        """
        Determine if the defender is out of field.
        """
        
        return (self.defender_pos.x <= -self.arena_range_x-0.2 or
                self.defender_pos.x >= self.arena_range_x+0.2 or
                self.defender_pos.y <= -self.arena_range_y-0.2 or
                self.defender_pos.y >= self.arena_range_y+0.2)

def main(args=None):
    rclpy.init(args=args)
    robot_pub = Defender()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()