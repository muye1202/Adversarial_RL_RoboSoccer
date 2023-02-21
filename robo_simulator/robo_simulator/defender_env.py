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
    arena_range_y = arena_range_x
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
        self.defender_pub = self.create_publisher(Info, "defend/defender_info", 10)

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
        self.last_dist_to_start = 0.0
    
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
        
    def attacker_to_ball_dist(self):
        
        return math.sqrt((self.ball_pos.x - self.robot_pos.x)**2 + 
                         (self.ball_pos.y - self.robot_pos.y)**2)
    
    def step(self):
        done = False
        rewards = 0.0

        self.count_steps += 1
        ### DEFENDER REWARDS ###
        # check whether the robot moves towards
        # the goal in the past episode:
        if self.is_scored():
            rewards -= 3000.0
            done = True

        # stop the defender when it runs pass the attacker
        if not self.is_scored() and self.is_out_of_range():
            rewards = -3000.0
            done = True

        if self.dist_between_players() <= 0.8:
            rewards += 500.0

        if (self.dist_between_players() <= 2.4 and abs(self.player_facing()) < 10.0):
            rewards += 500.0 * 15.0 / (self.player_facing() + 1)

        elif (self.dist_between_players() <= 2.4 and abs(self.player_facing()) < 15.0):
            rewards += -20*self.player_facing() + 600

        elif (self.dist_between_players() <= 2.4 and abs(self.player_facing()) > 30.0):
            rewards -= 50.0

        # if the player chases the opponent directly
        # it will obtain the max rewards
        if self.count_steps == 5:

            if (self.dist_between_players() <= 2.4 and self.chasing_score() < 5.0):
                if self.last_dist_to_start < self.dist_to_start():
                    rewards += 150.0
                else:
                    rewards += 100.0

            elif (self.dist_between_players() <= 2.4 and self.chasing_score() < 7.0):
                if self.last_dist_to_start < self.dist_to_start():
                    rewards += 70.0
                else:
                    rewards += 40.0

            elif (self.dist_between_players() <= 2.0 and self.chasing_score() > 15.0):
                rewards -= 50.0

            self.count_steps = 0
            self.last_defender_pos = self.defender_pos

        if self.ball_to_goal_dist() <= 2.0:
            rewards -= 500.0

        # punish if the player goes over the attacker
        # without being in range    
        if (self.defender_pos.x < self.robot_pos.x+0.1):
            done = True
            if (self.dist_between_players() >= 1.5):
                rewards -= 900.0

        step_info = Info()
        # dist_to_ball = math.sqrt((self.defender_pos.x - self.ball_pos.x)**2 + (self.defender_pos.y - self.ball_pos.y)**2)
        step_info.states = [self.defender_pos.x, self.defender_pos.y, self.defender_pos.theta, 
                            self.dist_between_players(), self.player_facing(), self.angle_between()]
        step_info.rewards = rewards
        step_info.done = done
        self.last_dist_players = self.dist_between_players()
        self.last_attacker_pos = self.robot_pos
        self.last_defender_dir = self.defender_pos.theta
        self.last_dist_to_start = self.dist_to_start()   # distance to the other side of the field

        self.defender_pub.publish(step_info)

    def angle_between(self):
        """
        Angle between players.
        """

        return math.degrees(math.atan2(self.robot_pos.y - self.defender_pos.y,
                                       self.robot_pos.x - self.defender_pos.x))

    def player_facing(self):
        """
        Is player facing the target (compare angles).
        """
        angle_between_players = self.angle_between()
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

    def dist_to_start(self):
        """
        Measure the Euclidean dist to start pos.
        """
        
        return abs(self.defender_pos.x - (-3.5))

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
        vect_to_opponent = np.array([self.robot_pos.x - 2.0,
                                     self.robot_pos.y - 0.0])
        vect_to_opponent = vect_to_opponent / np.linalg.norm(vect_to_opponent)

        vect_to_curr = np.array([self.defender_pos.x - 2.0,
                                 self.defender_pos.y - 0.0])
        vect_to_curr = vect_to_curr / np.linalg.norm(vect_to_curr)

        # normalized dot product
        dot_product = np.dot(vect_to_curr, vect_to_opponent)

        return math.degrees(np.arccos(dot_product))


def main(args=None):
    rclpy.init(args=args)
    robot_pub = Defender()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()