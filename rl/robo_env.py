"""
The environment for training 1 vs 1 robot soccer RL network.
"""
from infrastructure.agent import Agent
from infrastructure.world_model import WorldModel
from gym import Env
from gym.spaces import Box
import numpy as np


class RoboPlayer(Env):
    
    def __init__(self, role: str, agent: Agent, opponent: Agent, host, port, teamname, unnum=1):
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
            self.action_space = Box(low=np.array([0., -np.pi, 10., -55]), \
                                    high=np.array([100., np.pi, 40., 55]))
        
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
        
        # opponent agent
        self.opponent = opponent
        
        # role of the player
        self.side = role
        
        # see if the reset is for starting the match
        self.kick_off = True
        
        # store the last kick direction
        self.last_kick_dir = 0.
        
        # reset the simulation for different scenarios
        self.reset_flag = None
        
        # prevent goal kick
        self.prevent_goal_kick = False
        
        ## DEBUG FILE
        f = open('debug_train.txt', 'w')
        self.debug_file = f
        
    def kick_to_random_pos(self, player:Agent):
        # kick the ball to random position
        kick_dir = np.random.uniform(low=-45, high=45, size=1)
        kick_pow = np.random.uniform(low=0, high=50, size=1)
        player.wm.ah.kick(kick_pow, kick_dir)
    
    def is_ball_measurable(self, player:Agent):
        """
        Are the distance and dir of the ball accessible.
        """

        return (player.wm.ball.distance is not None \
                and player.wm.ball.direction is not None)
    
    def step(self, action):
        done = False
        is_nearer_to_goal = False
        rewards = 0
        
        # send cmd for opponent player
        if self.opponent.send_commands:
            self.opponent.send_commands = False
            self.opponent.wm.ah.send_commands()
        
        if self.opponent.should_think_on_data:
            
            self.opponent.should_think_on_data = False
            
            # turn the body of the opponent to ball
            if self.opponent.wm.ball is not None:
                if self.is_ball_measurable(self.opponent):
                    self.opponent.wm.turn_body_to_object(self.opponent.wm.ball)
        
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
            
            ###################### SENDING COMMANDS ########################
            #################################################################
            
            # the agent should find the ball if ball not in sight
            if self.agent.wm.ball is None or self.agent.wm.ball.direction is None:
                self.agent.wm.ah.turn(30)

            elif self.agent.wm.ball is not None and not self.agent.wm.is_before_kick_off():
                ## the player should move to the ball
                ## with a constant speed in order to 
                ## perform meaningful dribbling and kicking
                ## slow down the player as it approaches
                player_vel = 70
                speed_dir = 0.
                if self.agent.wm.ball is not None:
                    if self.is_ball_measurable(self.agent):
                        speed_dir = self.agent.wm.ball.direction
                        if self.agent.wm.ball.distance <= 5:
                            player_vel = 60

                self.agent.wm.ah.turn(speed_dir)
                self.agent.wm.ah.dash(player_vel, speed_dir)
                self.last_kick_dir = speed_dir

                kick_power = action[2]
                kick_dir = action[3]
                if self.agent.wm.is_ball_kickable():
                    self.agent.wm.ah.kick(kick_power, kick_dir)
            
            ############# GET UPDATES ###########
            ##########################################
                
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
            
                ############## STATES ################
                # update the state of the player and ball
                # used to determine if an episode has ended
                self.state = np.array([curr_pos[0], curr_pos[1]])
                
                # determine if the ball is in range
                # by checking if play_mode is drop ball
                ball_out_range = (self.agent.wm.is_kick_in() or self.agent.wm.is_goal_kick())

                self.debug_file.write("current play mode: " + str(self.agent.wm.play_mode) + '\n')

                # determine if the player comes nearer to the goal
                if len(goal_to_player) == 4:
                    is_nearer_to_goal = (goal_to_player[2] - goal_to_player[0] > 0)
                
                # determine if player is in shooting range
                is_in_shoot_range = False
                goal_dist = 50
                for goal in self.agent.wm.goals:
                    if goal.goal_id == enemy_goal_id:
                        goal_dist = goal.distance
                
                is_in_shoot_range = (goal_dist < 20.)

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
                
                # attacking side
                # NOTE: implement attacker reward 1 5 6 7 8.
                if self.agent.wm.is_scored():
                    rewards = 10
                    done = True
                    self.reset_flag = 'goal'
                    self.agent.wm.is_goal = False
                    self.opponent.wm.is_goal = False

                elif ball_out_range:
                    rewards = -5
                    self.reset_flag = 'not_in_range'
                    done = True

                elif is_nearer_to_goal:
                    rewards += 0.5

                elif is_in_shoot_range:
                    rewards += 2
                
        info = {}
        return self.state, rewards, done, info

    def reset(self):
        # reset the player to initial position
        # ONLY WHEN KICKING OFF THE MATCH
        print("reset agent: " + str(self.reset_flag))
        
        ball_start_pos_x, ball_start_pos_y = 0., 0.
        if self.kick_off:
            self.agent.connect(host=self.host, port=self.port,\
                            teamname=self.teamname, unnum=self.unnum, side=WorldModel.SIDE_L)

            self.agent.play()
            
            # move the opponent to position
            self.opponent.connect(host=self.host, port=self.port,\
                                teamname='enemy', unnum=self.unnum, side=WorldModel.SIDE_R)
            self.opponent.play()
            
            self.kick_off = False
            
        else:
            # this means agent scored and position
            # is automatically reset by the server
            # opponent needs to kick off
            if self.opponent.send_commands:
                self.opponent.send_commands = False
                self.opponent.wm.ah.send_commands()
        
            if self.opponent.should_think_on_data:
                self.opponent.should_think_on_data = False
                
                print("it can send cmd")
                if self.reset_flag == 'goal':
                    # move the opponent to ball
                    # ONLY AGENT CAN SEE THE BALL
                    self.opponent.wm.teleport_to_point((5, 0.))

                    if self.opponent.wm.ball is None:
                        self.opponent.wm.ah.turn(-170)
                        
                    self.kick_to_random_pos(self.opponent)
                    
                elif self.reset_flag == 'not_in_range':
                    # turn the opponent to ball
                    if self.opponent.wm.ball is None:
                        # if the agent can see the ball
                        if self.agent.wm.ball is not None:
                            if self.is_ball_measurable(self.agent):
                                # turn opponent to ball
                                ball_pos = self.agent.wm.get_object_absolute_coords(self.agent.wm.ball)
                                self.opponent.wm.turn_body_to_point(ball_pos)
                                self.opponent.wm.teleport_to_point(ball_pos)
                                
                    else:
                        ball_pos = self.opponent.wm.get_object_absolute_coords(self.opponent.wm.ball)
                        self.opponent.wm.turn_body_to_object(self.opponent.wm.ball)
                        self.opponent.wm.teleport_to_point(ball_pos)

                    # kick the ball to random position
                    self.kick_to_random_pos(self.opponent)

        self.state = np.array([ball_start_pos_x, ball_start_pos_y]).reshape(-1)
                    
        return self.state
        

