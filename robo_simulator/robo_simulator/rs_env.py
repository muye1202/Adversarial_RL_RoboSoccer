import rclpy
from rclpy.node import Node
import numpy as np
import math
from gym.spaces import Box
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Empty
from rl_interfaces.msg import Info


class RoboPlayer(Node):
    
    arena_range_x = 3.5
    arena_range_y = arena_range_x/2
    robot_pos = Pose2D()
    ball_pos = Point()

    def __init__(self, role="attacker"):
        super().__init__("robo_player")
        # create action space for the agent
        # each agent can do the following things:
        # moving speed (velocity), moving direction (turn Moment),
        # shooting power and direction.
        # If defender, add in tackle action (tackle power).
        # For details:
        # https://rcsoccersim.readthedocs.io/en/latest/soccerclient.html#control-commands
        if role == "defender":
            # move_speed move_dir tackle
            self.action_space = Box(low=np.array([0., -np.pi, -90.]), \
                                    high=np.array([100., np.pi, 90.]))
        else:
            # [move_speed move_dir kick_pow kick_dir]
            # the kick direction needs to be within view angle
            self.action_space = Box(low=np.array([30., -90]), \
                                    high=np.array([70., 90]))
        
        # create state space of the agent
        # the state space of the agent includes:
        # x and y position, which needs to be calculated 
        # by the visual info coming from visual sensor
        self.observation_space = Box(low=np.array([-50., -35.]), high=np.array([50., 35.]), dtype=np.float64)
        
        # state dict: {position of the player, pos of the ball}
        self.state = np.array([0., 0.])
        
        # last ball position
        self.last_ball_dist = self.arena_range_x
        self.last_ball_dist_y = 0.
        
        #######################################
        timer_period = 0.001
        self.timer_ = self.create_timer(timer_period, self.timer_callback)

        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "field/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "field/robot_pos", self.robot_callback, 10)
        self.last_dir = 0.
        
        # publish step function update
        self.step_update_pub = self.create_publisher(Info, "rs_simulator/step_info", 1)
        self.step_info = Info()
        
        # see if new cmd has been sent
        self.state_sub = self.create_subscription(Empty, "~/update", self.update_callback, 10)
        self.step_state = False
        
        # count how many steps have been taken
        self.count_steps = 0.
    
    def timer_callback(self):
        if self.step_state:
            self.step()
            
            self.step_state = False
    
    def update_callback(self, _):
        self.step_state = True
    
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
        
        # self.get_logger().info("robot position: " + str(self.robot_pos.x) + " " + str(self.robot_pos.y))
        ### REWARDS ###
        # For attackers:
        #       1. For each step without scoring, reward = 0
        #       2. Lose the control of the ball, reward = -10
        #       3. Shooting route blocked by defender, reward = -1
        #       4. Find a clear route to goal, reward = +5
        #       5. Score a goal, reward is set to 10
        #       6. The ball's distance towards the goal advances, reward = + 0.5
        #       7. The ball is out of field, reward = -5
        #       8. Inside shooting range to the goal, reward = + 2
        # For defenders:
        #       For each step without being scored, reward = +1
        #       Taking control over the ball, reward = +10
        #       Blocking shooting route, reward = +1
        
        # check whether the robot moves towards
        # the goal in the past episode:
        if self.last_ball_dist - self.ball_to_goal_dist() > 0.01:    
            rewards += 1.0
            
            # self.get_logger().info("to goal dist decreased: " + str(self.ball_to_goal_dist()-self.last_ball_dist))
            # self.last_ball_dist = self.ball_to_goal_dist()

        elif self.last_ball_dist - self.ball_to_goal_dist() < 0.01:
            rewards -= 1.0
            
            # self.get_logger().info("to goal dist increased: " + str(self.ball_to_goal_dist()-self.last_ball_dist))
            # self.last_ball_dist = self.ball_to_goal_dist()
            
        #The robot should be rewarded if it faces the goal direction
        if self.is_player_facing_goal():
            rewards += 0.5
            
        # check whether the robot is advancing to the goal
        if self.is_scored():
            rewards = 20.
            done = True
            
        if not self.is_scored() and self.is_dead_ball():
            if rewards == 0.:
                rewards = -2.0
            else:
                rewards -= 0.5
            
            done = True

        self.step_info.states = [self.ball_pos.x, self.ball_pos.y]
        self.step_info.rewards = rewards
        self.step_info.done = done
        
        self.step_update_pub.publish(self.step_info)
        self.last_dir = self.robot_pos.theta
        self.last_ball_dist = self.ball_to_goal_dist()
            

    def render(self):
        pass
    
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

def main(args=None):
    rclpy.init(args=args)
    robot_pub = RoboPlayer()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()