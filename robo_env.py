"""
The environment for training 1 vs 1 robot soccer RL network.
"""
from . import world_model as wm
from . import handler
from .agent import Agent
from gym import Env
from gym.spaces import Box
import numpy as np


class RoboPlayer(Env):
    
    def __init__(self, side: str, agent: Agent):
        # create action space for the agent
        # each agent can do the following things:
        # moving speed (velocity), moving direction (turn Moment),
        # and turn_neck_angle. If defender, add in tackle action (tackle power).
        # For details:
        # https://rcsoccersim.readthedocs.io/en/latest/soccerclient.html#control-commands
        if side == "defender":
            #                                     speed move_dir neck_dir tackle
            self.action_space = Box(low=np.array([-100, -np.pi, -np.pi/2, -90]), \
                                    high=np.array([100, np.pi, np.pi/2, 90]), dtype=float)
        else:
            self.action_space = Box(low=np.array([-100, -np.pi, -np.pi/2]), \
                                    high=np.array([100, np.pi, np.pi/2]), dtype=float)
        
        # create state space of the agent
        # the state space of the agent includes:
        # x and y position, which needs to be calculated 
        # by the visual info coming from visual sensor
        self.observation_space = Box(low=np.array([-50, -35]), high=np.array([50, 35]), dtype=float)
        
        # the learning agent
        # we are controlling this agent
        self.agent = agent
        
        # side of the player
        self.side = side
        
    def step(self, action):
        #### update the agent by sending every command to the server ####
        # Get current score
        if self.agent.wm.side == "l":
            score = self.agent.wm.score_l
        else:
            score = self.agent.wm.score_r
        # the player sprints towards certain dir with speed
        dash_speed = action[0]
        dash_dir = action[1]
        self.agent.wm.ah.dash(dash_speed, dash_dir)

        # the player turns their neck to observe
        neck_ang = action[2]
        self.agent.wm.ah.turn_neck(neck_ang)

        if self.side == "defender":
            # send the tackle command
            tackle = action[3]
            self.agent.wm.ah.tackle(tackle, "off")
        
        ### Get the sensor updates ###
        # get current score
        if self.agent.wm.side == "l":
            curr_score = self.agent.wm.score_l
        else:
            curr_score = self.agent.wm.score_r
            
        # get current position
        # TODO: calculate current position
        
        # TODO: get the object in view
        
        
        ### REWARDS ###
        rewards = 0
        # TODO: For attackers:
        #       For each step without scoring, reward = -1
        #       Lose the control of the ball, reward = -10
        #       Shooting route blocked by defender, reward = -1
        #       Find a clear route to goal, reward = +5
        #       Score a goal, reward = +10
        # TODO: For defenders:
        #       For each step without being scored, reward = +1
        #       Taking control over the ball, reward = +10
        #       Blocking shooting route, reward = +1
        # TODO: use the self.agent.wm.score_l or r to access score
        # attacking side
        if self.side == "attacker":
            return 
        
