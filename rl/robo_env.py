"""
The environment for training 1 vs 1 robot soccer RL network.
"""
from infrastructure.agent import Agent
from infrastructure.world_model import WorldModel
from gym import Env
from gym.spaces import Box
import numpy as np
import time


class RoboPlayer(Env):
    
    def __init__(self, role: str, agent: Agent, host, port, teamname, unnum):
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
            #                                     move_speed move_dir tackle
            self.action_space = Box(low=np.array([-100., -np.pi, -90.]), \
                                    high=np.array([100., np.pi, 90.]))
        else:
                                                # move_speed move_dir kick_pow kick_dir
            self.action_space = Box(low=np.array([-100., -np.pi, -100., -np.pi]), \
                                    high=np.array([100., np.pi, 100., np.pi]))
        
        # create state space of the agent
        # the state space of the agent includes:
        # x and y position, which needs to be calculated 
        # by the visual info coming from visual sensor
        self.observation_space = Box(low=np.array([-50., -35.]), high=np.array([50., 35.]))
        
        # state dict: {position of the player, pos of the ball}
        self.state = np.array([0., 0.])   # {"ball": np.array([0., 0.]), "player": np.array([0., 0.])}
        
        # the learning agent
        # we are controlling this agent
        self.agent = agent
        
        # role of the player
        self.side = role
        
    def step(self, action):
        done = False
        is_nearer_to_goal = False
        rewards = 0
        
        if self.agent.send_commands:
            self.agent.send_commands = False
            self.agent.wm.ah.send_commands()
        
        if self.agent.should_think_on_data:
            
            self.agent.should_think_on_data = False
            
            # get the ball's distance to enemy goal:
            ## get enemy goal id
            if self.agent.wm.side == "l":
                enemy_goal_id = "r"
            else:
                enemy_goal_id = "l"
            
            goal_to_player = []
            if self.agent.wm.ball is not None:
                for goal in self.agent.wm.goals:
                    if goal.goal_id == enemy_goal_id:
                        goal_to_player.append(goal.distance)
                        goal_to_player.append(goal.direction)
            
            #### SENDING COMMANDS ####
            #################################################################
            # the player sprints towards certain dir with speed
            dash_speed = action[0]
            dash_dir = action[1]
            
            # the agent should find the ball if ball not in sight
            if self.agent.wm.ball is None or self.agent.wm.ball.direction is None:
                self.agent.wm.ah.turn(30)
            
            # The player needs to move to the ball
            if self.agent.wm.ball is not None and not self.agent.wm.is_before_kick_off():
                ## the player should face the ball
                if self.agent.wm.ball.direction is not None:
                    self.agent.wm.ah.turn(self.agent.wm.ball.direction / 2)
                
                ## the player should move to the ball
                self.agent.wm.ah.dash(dash_speed, dash_dir)

                if self.side == "defender":
                    # send the tackle command
                    tackle = action[3]
                    self.agent.wm.ah.tackle(tackle, "off")
                else:
                    # the player kicks the ball with pow and dir
                    kick_power = action[2]
                    kick_dir = action[3]
                    
                    if self.agent.wm.is_ball_kickable():
                        self.agent.wm.ah.kick(kick_power, kick_dir)
            
            # # DO NOT move on until new msg from server has been received
            # while not self.agent.received_update():
            #     time.sleep(0.0001)

            ### Get the sensor updates ###
            ##############################
            # get current score
            if self.agent.wm.side == "l":
                curr_score = self.agent.wm.score_l
                enemy_score = self.agent.wm.score_r
            else:
                curr_score = self.agent.wm.score_r
                enemy_score = self.agent.wm.score_l
                
            # get current position tuple: ( , )
            curr_pos = self.agent.wm.abs_coords
            
            ## get enemy goal distance and dir relative to player
            ## [dist, dir]
            for goal in self.agent.wm.goals:
                if goal.goal_id == enemy_goal_id:
                    goal_to_player.append(goal.distance)
                    goal_to_player.append(goal.direction)
                    
            ## get dist and dir of the ball to player
            ## [dist, dir]
            ## UPDATE ONLY BALL IS NOT NONE
            if self.agent.wm.ball is not None:
                ball_to_player = []
                ball_to_player.append(self.agent.wm.ball.distance)
                ball_to_player.append(self.agent.wm.ball.direction)
            
                ### STATES ###

                # update the state of the player and ball
                # used to determine if an episode has ended
                self.state = np.array([curr_pos[0], curr_pos[1]])
                
                # use the abs coord of player to get abs coord
                # of the ball
                # ball_abs_coord_x = curr_pos[0] + ball_to_player[0]*np.cos(ball_to_player[1])
                # ball_abs_coord_y = curr_pos[1] + ball_to_player[0]*np.sin(ball_to_player[1])
                # self.state["ball"] = np.array([ball_abs_coord_x, ball_abs_coord_y])
                
                # determine if the ball is in range
                # by checking if play_mode is drop ball
                ball_in_range = (self.agent.wm.play_mode == WorldModel.PlayModes.DROP_BALL)
                
                # determine if the player comes nearer to the goal
                if len(goal_to_player) == 4:
                    is_nearer_to_goal = (goal_to_player[2] - goal_to_player[0] > 0)
                
                ### REWARDS ###
                # For attackers:
                #       1. For each step without scoring, reward = 0
                #       2. Lose the control of the ball, reward = -10
                #       3. Shooting route blocked by defender, reward = -1
                #       4. Find a clear route to goal, reward = +5
                #       5. Score a goal, reward is set to 10
                #       6. The ball's distance towards the goal advances, reward = + 0.5
                #       7. The ball is out of field, reward = -5
                # For defenders:
                #       For each step without being scored, reward = +1
                #       Taking control over the ball, reward = +10
                #       Blocking shooting route, reward = +1
                
                # attacking side
                # NOTE: implement attacker reward 1 5 6 7.
                if self.side == "attacker":
                    if curr_score > enemy_score:
                        rewards = 10
                        done = True
                    elif not ball_in_range:
                        rewards = -5
                    elif curr_score < enemy_score:
                        rewards = -10
                        done = True
                    elif is_nearer_to_goal:
                        rewards += 0.5
                
        info = {}
        return self.state, rewards, done, info

    def reset(self):
        # reset the player to initial position
        # by disconnect and re-connect
        print("reset agent")
        self.agent.disconnect()
        self.agent.connect(host=self.host, port=self.port,\
                           teamname=self.teamname, unnum=self.unnum, side=WorldModel.SIDE_L)

        self.agent.play()

