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
        # last player position
        self.last_attacker_pos = Pose2D()
        self.last_attacker_pos.x = 0.0
        self.last_attacker_pos.y = 0.0

        self.last_defender_pos = Pose2D()
        self.last_defender_pos.x = 2.0
        self.last_defender_pos.y = 0.0
        
        #######################################
        timer_period = 0.002
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
        self.count_steps = 1.0
        
        # last dist between players
        self.last_dist_players = 0.0
        self.last_x_dist = 2.0   # last x dist between players
        self.last_y_dist = 0.0   # last y dist between players
        self.last_defender_dir = 0.0
    
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

        in_tackle_range = False
        ### DEFENDER REWARDS ###
        # For each step without being scored, reward = +1
        # Taking control over the ball, reward = +10
        # Blocking shooting route, reward = +1

        # check whether the robot moves towards
        # the goal in the past episode:
        if self.is_scored():
            rewards -= 3.0
            done = True

        # stop the defender when it runs pass the attacker
        if not self.is_scored() and self.is_out_of_range():
            rewards = -1.0
            done = True
            
        elif self.defender_pos.x < 1.0+self.robot_pos.x:
            done = True

        if self.dist_between_players() <= 0.8 and self.is_player_facing(role="attacker", del_angle=2.0):
            rewards += 2.0
            done = True
            
            in_tackle_range = True

        elif self.dist_between_players() <= 1.6 and self.is_player_facing(role="attacker", del_angle=5.0):
            rewards += 1.0
            done = True

            if (abs(self.defender_pos.theta - self.last_defender_dir) <= 90.0):
                rewards += 0.1

        # if the player chases the opponent directly
        # it will obtain the max rewards
        if abs(self.robot_pos.x - self.defender_pos.x) <= 3.2 and self.chasing_score() >= 0.97:
            rewards += 2.0
            
        elif abs(self.robot_pos.x - self.defender_pos.x) <= 3.2 and self.chasing_score() >= 0.95:
            rewards += 0.5

        if self.count_steps >= 20 and not in_tackle_range:
            if rewards > 0:
                rewards *= 0.5
            else:
                rewards -= 3.0
                

        step_info = Info()
        step_info.states = [self.defender_pos.x, self.defender_pos.y, self.defender_pos.theta]
        step_info.rewards = rewards
        step_info.done = done
        self.last_dist_players = self.dist_between_players()
        self.last_attacker_pos = self.robot_pos
        self.last_defender_pos = self.defender_pos
        self.count_steps += 1
        self.last_defender_dir = self.defender_pos.theta

        self.defender_pub.publish(step_info)

    def angle_between(self):
        """
        Angle between players.
        """
        
        return math.degrees(math.atan2(self.robot_pos.y - self.defender_pos.y,
                                       self.robot_pos.x - self.defender_pos.x))

    def is_player_facing(self, role, del_angle):
        """
        Is player facing the target (compare angles).
        """

        # check if defender can "see" attacker
        angle_between_players = self.angle_between()
        def_facing = math.degrees(self.defender_pos.theta)
        attacker_facing = math.degrees(self.robot_pos.theta)

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
        
        return math.sqrt((self.robot_pos.x - self.defender_pos.x)**2
                + (self.robot_pos.y - self.defender_pos.y)**2)

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
        
        Note: the episode will end if the defender run pass the attacker.
        """

        return (self.defender_pos.x <= -self.arena_range_x-0.2 or
                self.defender_pos.x >= self.arena_range_x+0.2 or
                self.defender_pos.y <= -self.arena_range_y-0.2 or
                self.defender_pos.y >= self.arena_range_y+0.2)

    def chasing_score(self):
        """
        Check how much does the player advance in the direction of the opponent.
        
        Calculate the dot product between the vector from the player to the opponent,
        and the vector from its last to current position; then assign rewards proportional
        to the dot product
        """
        vect_to_opponent = np.array([self.robot_pos.x - self.defender_pos.x, 
                                     self.robot_pos.y - self.defender_pos.y])

        vect_to_curr = np.array([self.defender_pos.x - self.last_defender_pos.x,
                                 self.defender_pos.y - self.last_defender_pos.y])
        dot_product = np.dot(vect_to_curr, vect_to_opponent)

        return dot_product / (np.linalg.norm(vect_to_curr)*np.linalg.norm(vect_to_opponent))


def main(args=None):
    rclpy.init(args=args)
    robot_pub = Defender()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()