import rclpy
from rclpy.node import Node
import numpy as np
import math
from gym.spaces import Box
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Empty
from rl_interfaces.msg import Info


class RoboPlayer(Node):
    
    arena_size_x = 5.5
    arena_size_y = 5.5/2
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
        self.last_ball_dist = self.arena_size_x
        
        #######################################
        timer_period = 0.01
        self.timer_ = self.create_timer(timer_period, self.timer_callback)
        
        # publish reset command to the simulator
        self.reset_pub = self.create_publisher(Empty, "~/reset_flag", 10)

        # receive ball and robot position
        self.ball_sub = self.create_subscription(Point, "field/ball_pos", self.ball_callback, 10)
        self.robot_sub = self.create_subscription(Pose2D, "field/robot_pos", self.robot_callback, 10)
        
        # publish step function update
        self.step_update_pub = self.create_publisher(Info, "rs_simulator/step_info", 1)
        self.step_info = Info()
        
        # see if new cmd has been sent
        self.state_sub = self.create_subscription(Empty, "~/update", self.update_callback, 10)
        self.step_state = False
    
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

        # check whether the robot is advancing to the goal
        if self.ball_to_goal_dist() < self.last_ball_dist:
            rewards += 0.5
            self.last_ball_pos = self.ball_to_goal_dist()

        if self.ball_to_goal_dist() > self.last_ball_dist:
            rewards -= 1.0
            self.last_ball_pos = self.ball_to_goal_dist()
            
        if self.is_dead_ball():
            rewards -= 5
            done = True
            
        if self.is_scored():
            rewards += 10
            done = True
        
        self.step_info.states = [self.robot_pos.x, self.robot_pos.y]
        self.step_info.rewards = rewards
        self.step_info.done = done
        
        self.step_update_pub.publish(self.step_info)
        if done:
            self.reset()
            self.get_logger().info("RESET!")

    def render(self):
        pass

    def reset(self):
        self.reset_signal()
        
        # # wait until position is reset
        # current_robo_pos = self.robot_pos
        # while current_robo_pos.x >= 1e-3 and current_robo_pos.y >= 1e-3:
        #     pass
        
        # self.step_info.states = [current_robo_pos.x, current_robo_pos.y]
    
    def is_scored(self):
        """
        Check whether the attacking side has scored.
        """
        
        return (self.ball_pos.x >= 0.5*self.arena_size_x-0.12 and
                self.ball_pos.y <= 0.1*self.arena_size_y-0.12 and
                self.ball_pos.y >= 0.1*self.arena_size_y+0.12)
        
    def is_dead_ball(self):
        """
        Check whether the ball is out of range.
        """
        
        return (self.ball_pos.x <= -self.arena_size_x-0.1 or
                self.ball_pos.x >= self.arena_size_x+0.1 or
                self.ball_pos.y <= -self.arena_size_y-0.1 or
                self.ball_pos.y >= self.arena_size_y+0.1)
        
    def reset_signal(self):
        """
        Send the reset signal to simulator to reset the position
        of the robot and the ball.
        """
        
        self.reset_pub.publish(Empty())
    
    def ball_to_goal_dist(self):
        """
        Calculate the distance between the ball and the goal
        """
        
        return math.sqrt((self.ball_pos.x - self.arena_size_x)**2 + 
                         (self.ball_pos.y - self.arena_size_y)**2)

def main(args=None):
    rclpy.init(args=args)
    robot_pub = RoboPlayer()
    rclpy.spin(robot_pub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()