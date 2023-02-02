"""
The environment for training 1 vs 1 robot soccer RL network.

FOR THE SOCCER SERVER SIMULATOR
"""
from infrastructure.agent import Agent
from infrastructure.world_model import WorldModel
from gym import Env
from gym.spaces import Box
import numpy as np


class RoboPlayer(Env):
    
    def __init__(self, role: str, agent: Agent, opponent: Agent, helper: Agent,
                 host, port, teamname, unnum=1):
        # variables needed for reset()
        self.host = host
        self.port = port
        self.teamname = teamname
        self.unnum = unnum

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
            self.action_space = Box(low=np.array([0., -np.pi, 20., -55]), \
                                    high=np.array([100., np.pi, 30., 55]))
        
        # create state space of the agent
        # the state space of the agent includes:
        # x and y position, which needs to be calculated 
        # by the visual info coming from visual sensor
        self.observation_space = Box(low=np.array([-50., -35.]), high=np.array([50., 35.]), dtype=np.float64)
        
        # state dict: {position of the player, pos of the ball}
        self.state = np.array([0., 0.])   # {"ball": np.array([0., 0.]), "player": np.array([0., 0.])}
        
        # the learning agent
        # we are controlling this agent
        self.agent = agent
        
        # opponent agent
        self.opponent = opponent
        
        self.helper = helper
        
        # role of the player
        self.side = role
        
        # store the last kick direction
        self.last_kick_dir = 0.
        
        # reset the simulation for different scenarios
        self.reset_flag = None
        
        # prevent goal kick
        self.prevent_goal_kick = False
        
        self.reset_flag = None
        
        self.ball_pos = None
    
    def step(self, action):
        done = False
        rewards = self.agent.rewards
        
        self.reset_flag, rewards, done = self.agent.think(action=action, done=done, rewards=rewards, role="agent")
        self.opponent.think(action=action, done=done, rewards=rewards, role="trainer")
        self.helper.think(action=action, done=done, rewards=rewards, role="helper")
        
        info = {}
        return self.state, rewards, done, info

    def render(self):
        pass

    def reset(self):
        # reset the player to initial position
        # ONLY WHEN KICKING OFF THE MATCH
        if self.agent.kick_off:
            self.agent.connect(host=self.host, port=self.port,\
                            teamname=self.teamname, unnum=self.unnum, side=WorldModel.SIDE_L)

            self.agent.play()
            self.opponent.connect(host=self.host, port=self.port,\
                                teamname='enemy', unnum=self.unnum,\
                                side=WorldModel.SIDE_R, goalie=True)
            self.opponent.play()
            
        elif not self.agent.kick_off:
            self.opponent.reset_flag = self.reset_flag
            self.opponent.trainer()
            self.helper.pass_to_player(side='l', unum=1)
            self.agent.rewards = 0.
            self.agent.done = False

        agent_pos = self.agent.wm.abs_coords
        self.state = np.array([agent_pos[0], agent_pos[1]])

        return self.state


