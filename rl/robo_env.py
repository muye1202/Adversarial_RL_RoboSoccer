"""
The environment for training 1 vs 1 robot soccer RL network.
"""
from infrastructure.agent import Agent
from gym import Env
from gym.spaces import Box
import numpy as np


class RoboPlayer(Env):
    
    def __init__(self, side: str, agent: Agent):
        # create action space for the agent
        # each agent can do the following things:
        # moving speed (velocity), moving direction (turn Moment),
        # shooting power and direction.
        # If defender, add in tackle action (tackle power).
        # For details:
        # https://rcsoccersim.readthedocs.io/en/latest/soccerclient.html#control-commands
        if side == "defender":
            #                                     move_speed move_dir tackle
            self.action_space = Box(low=np.array([-100, -np.pi, -90]), \
                                    high=np.array([100, np.pi, 90]), dtype=float)
        else:
                                                # move_speed move_dir kick_pow kick_dir
            self.action_space = Box(low=np.array([-100, -np.pi, -100, -np.pi]), \
                                    high=np.array([100, np.pi, 100, np.pi]), dtype=float)
        
        # create state space of the agent
        # the state space of the agent includes:
        # x and y position, which needs to be calculated 
        # by the visual info coming from visual sensor
        self.observation_space = Box(low=np.array([-50, -35]), high=np.array([50, 35]), dtype=float)
        
        # state dict: {position of the player, pos of the ball}
        self.state = {"ball": np.array([0., 0.]), "player": np.array([0., 0.])}
        
        # the learning agent
        # we are controlling this agent
        self.agent = agent
        
        # side of the player
        self.side = side
        
    def step(self, action):
        done = False
        rewards = 0
        #### update the agent by sending every command to the server ####
        #################################################################
        # Get current score
        if self.agent.wm.side == "l":
            score = self.agent.wm.score_l
        else:
            score = self.agent.wm.score_r

        # the player sprints towards certain dir with speed
        dash_speed = action[0]
        dash_dir = action[1]

        ## the player should face the ball
        self.agent.wm.ah.turn(self.agent.wm.ball.direction / 2)
        self.agent.wm.ah.dash(dash_speed, dash_dir)

        if self.side == "defender":
            # send the tackle command
            tackle = action[3]
            self.agent.wm.ah.tackle(tackle, "off")
        else:
            # the player kicks the ball with pow and dir
            kick_power = action[2]
            kick_dir = action[3]
            self.agent.wm.ah.kick(kick_power, kick_dir)
        
        # DO NOT move on until new msg from server has been received
        while not self.agent.received_update():
            ### Get the sensor updates ###
            ##############################
            # get current score
            if self.agent.wm.side == "l":
                curr_score = self.agent.wm.score_l
            else:
                curr_score = self.agent.wm.score_r
                
            # get current position tuple: ( , )
            curr_pos = self.agent.wm.abs_coords
            
            # get the ball's distance to enemy goal:
            ## get enemy goal id
            if self.agent.wm.side == "l":
                enemy_goal_id = "r"
            else:
                enemy_goal_id = "l"
            
            ## get enemy goal distance and dir relative to player
            ## [dist, dir]
            goal_to_player = []
            for goal in self.agent.wm.goals:
                if goal.goal_id == enemy_goal_id:
                    goal_to_player.append(goal.distance)
                    goal_to_player.append(goal.direction)
                    
            ## get dist and dir of the ball to player
            ## [dist, dir]
            ball_to_player = []
            ball_to_player.append(self.agent.wm.ball.distance)
            ball_to_player.append(self.agent.wm.ball.direction)
            
            ### STATES ###

            # update the state of the player and ball
            # used to determine if an episode has ended
            self.state["player"] = np.array([curr_pos[0], curr_pos[1]])
            # use the abs coord of player to get abs coord
            # of the ball
            ball_abs_coord_x = curr_pos[0] + ball_to_player[0]*np.cos(ball_to_player[1])
            ball_abs_coord_y = curr_pos[1] + ball_to_player[0]*np.sin(ball_to_player[1])
            self.state["ball"] = np.array([ball_abs_coord_x, ball_abs_coord_y])
            
            ### REWARDS ###
            # For attackers:
            #       1. For each step without scoring, reward = 0
            #       2. Lose the control of the ball, reward = -10
            #       3. Shooting route blocked by defender, reward = -1
            #       4. Find a clear route to goal, reward = +5
            #       5. Score a goal, reward is set to 10
            #       6. The ball's distance towards the goal advances, reward = + 0.5
            # For defenders:
            #       For each step without being scored, reward = +1
            #       Taking control over the ball, reward = +10
            #       Blocking shooting route, reward = +1
            
            # attacking side
            # NOTE: implement attacker reward 1 and 5.
            if self.side == "attacker":
                if curr_score > score:
                    rewards = 10
                
        info = {}
        return self.state, rewards, done, info

    def reset(self):
        # reset the player to initial position
        # by disconnect and re-connect
        self.agent.disconnect()
        self.agent.connect()
