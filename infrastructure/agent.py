#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent/infrastructure')
import threading
import time
import sock
import sp_exceptions
import handler
import world_model
import numpy as np

class Agent:
    def __init__(self):
        # whether we're connected to a server yet or not
        self.__connected = False

        # the socket used to communicate with the server
        self.__sock = None

        # models and the message handler for parsing and storing information
        self.wm = None
        self.msg_handler = None
        
        # store the msg from the server
        self.recv_msg = None

        # parse thread and control variable
        self.__parsing = False
        self.__msg_thread = None

        self.__thinking = False # think thread and control variable
        self.__think_thread = None

        # whether we should run the think method
        self.should_think_on_data = False

        # whether we should send commands
        self.send_commands = False
        
        # whether updates from server is obtained
        self.recv_message = False
        
        self.in_kick_off_formation = False
        
        # interfacing with training loop
        self.rewards = 0.
        self.done = False
        self.process_done = False
        self.reset_flag = None
        self.is_in_right_side = True
        self.trainer_count = 0
        
        # see if the reset is for starting the match
        self.kick_off = True

    def connect(self, host, port, teamname, unnum, side, goalie=False, version=17):
        """
        Gives us a connection to the server as one player on a team.  This
        immediately connects the agent to the server and starts receiving and
        parsing the information it sends.
        """

        # if already connected, raise an error since user may have wanted to
        # connect again to a different server.
        if self.__connected:
            msg = "Cannot connect while already connected, disconnect first."
            raise sp_exceptions.AgentConnectionStateError(msg)

        # the pipe through which all of our communication takes place
        # self.__sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__sock = sock.Socket(host, port)

        # our models of the world and our body
        self.wm = world_model.WorldModel(handler.ActionHandler(self.__sock))

        # set the team name of the world model to the given name
        self.wm.teamname = teamname
        
        # set uniform number
        self.wm.uniform_number = unnum

        self.wm.side = side

        # handles all messages received from the server
        self.msg_handler = handler.MessageHandler(self.wm)

        # set up our threaded message receiving system
        self.__parsing = True # tell thread that we're currently running
        self.__msg_thread = threading.Thread(target=self.__message_loop,
                name="message_loop")
        self.__msg_thread.daemon = True # dies when parent thread dies
        
        # # TODO: fix the __message__loop threading
        # #       call the function instead
        # self.__message_loop()

        # start processing received messages. this will catch the initial server
        # response and all subsequent communication.
        self.__msg_thread.start()

        # send the init message and allow the message handler to handle further
        # responses.
        if not goalie:
            init_address = self.__sock.address
            init_msg = "(init %s (version %d))"
            init_msg = (init_msg % (teamname, version))
            self.__sock.send(init_msg)
        else:
            init_address = self.__sock.address
            init_msg = "(init %s (version %d) (goalie))"
            init_msg = (init_msg % (teamname, version))
            self.__sock.send(init_msg)

        # wait until the socket receives a response from the server and gets its
        # assigned port.
        while self.__sock.address == init_address:
            time.sleep(0.0001)

        ##### NOT NEEDED DURING TRAINING ###
        ## create our thinking thread.  this will perform the actions necessary
        ## to play a game of robo-soccer.
        # self.__thinking = False
        # self.__think_thread = threading.Thread(target=self.__think_loop,
        #         name="think_loop")
        # self.__think_thread.daemon = True

        # set connected state.  done last to prevent state inconsistency if
        # something goes wrong beforehand.
        self.__connected = True
        
        self.setup_environment()

    def play(self):
        """
        USED FOR TRAINING TO START MATCH

        Kicks off the thread that does the agent's thinking, allowing it to play
        during the game.  Throws an exception if called while the agent is
        already playing.
        """

        # ensure we're connected before doing anything
        if not self.__connected:
            msg = "Must be connected to a server to begin play."
            raise sp_exceptions.AgentConnectionStateError(msg)

        # throw exception if called while thread is already running
        if self.__thinking:
            raise sp_exceptions.AgentAlreadyPlayingError(
                "Agent is already playing.")

        # run the method that sets up the agent's persistant variables
        self.setup_environment()

        ##### NOT NEEDED DURING TRAINING ###
        # tell the thread that it should be running, then start it
        self.__thinking = True
        self.should_think_on_data = True
        # self.__think_thread.start() TODO: __think__thread is not starting
        self.__think_loop()

    def disconnect(self):
        """
        Tell the loop threads to stop and signal the server that we're
        disconnecting, then join the loop threads and destroy all our inner
        methods.

        Since the message loop thread can conceiveably block indefinitely while
        waiting for the server to respond, we only allow it (and the think loop
        for good measure) a short time to finish before simply giving up.

        Once an agent has been disconnected, it is 'dead' and cannot be used
        again.  All of its methods get replaced by a method that raises an
        exception every time it is called.
        """

        # don't do anything if not connected
        if not self.__connected:
            return

        # tell the loops to terminate
        self.__parsing = False
        self.__thinking = False

        # tell the server that we're quitting
        self.__sock.send("(bye)")

        ## NOT NEEDED until multithreading is fixed
        # # tell our threads to join, but only wait breifly for them to do so.
        # # don't join them if they haven't been started (this can happen if
        # # disconnect is called very quickly after connect).
        # if self.__msg_thread.is_alive():
        #     self.__msg_thread.join(0.01)

        # if self.__think_thread.is_alive():
        #     self.__think_thread.join(0.01)

        # reset all standard variables in this object.  self.__connected gets
        # reset here, along with all other non-user defined internal variables.
        Agent.__init__(self)

    def __message_loop(self):
        """
        Handles messages received from the server.

        This SHOULD NOT be called externally, since it's used as a threaded loop
        internally by this object.  Calling it externally is a BAD THING!
        """
        # loop until we're told to stop
        while self.__parsing:
            # receive message data from the server and pass it along to the
            # world model as-is.  the world model parses it and stores it within
            # itself for perusal at our leisure.
            raw_msg = self.__sock.recv(100000)
            raw_msg = (raw_msg.decode()).rstrip('\x00')
            
            # update the states of the players
            msg_type = self.msg_handler.handle_message(raw_msg)
            
            # we send commands all at once every cycle, ie. whenever a
            # 'sense_body' command is received
            if msg_type == handler.ActionHandler.CommandType.SENSE_BODY:
                self.send_commands = True
                self.recv_message = True

            # flag new data as needing the think loop's attention
            self.should_think_on_data = True
            
    def received_update(self):
        
        return self.recv_message

    def __think_loop(self):
        """
        Performs world model analysis and sends appropriate commands to the
        server to allow the agent to participate in the current game.

        Like the message loop, this SHOULD NOT be called externally.  Use the
        play method to start play, and the disconnect method to end it.
        """
        while self.__thinking:
            # tell the ActionHandler to send its enqueued messages if it is time
            if self.send_commands:
                self.send_commands = False
                self.wm.ah.send_commands()

            # only think if new data has arrived
            if self.should_think_on_data:
                # flag that data has been processed.  this shouldn't be a race
                # condition, since the only change would be to make it True
                # before changing it to False again, and we're already going to
                # process data, so it doesn't make any difference.
                self.should_think_on_data = False

                # performs the actions necessary for the agent to play soccer
                self.move_to_formation()
                
                robot_pos = self.wm.abs_coords
                if robot_pos != (None, None):
                    self.__thinking = False
                
            else:
                # prevent from burning up all the cpu time while waiting for data
                time.sleep(0.0001)

    def setup_environment(self):
        """
        Called before the think loop starts, this allows the user to store any
        variables/objects they'll want access to across subsequent calls to the
        think method.
        """

        self.in_kick_off_formation = False

    def move_to_formation(self):
        """
        USED FOR TRAINING: move to start position
        """
        if not self.in_kick_off_formation:

            # used to flip x coords for other side
            side_mod = -1
            if self.wm.side == world_model.WorldModel.SIDE_L:
                self.wm.teleport_to_point((5 * side_mod, 0))
                self.in_kick_off_formation = True
            else:
                side_mod = 1
                self.wm.teleport_to_point((5, 10))
                self.in_kick_off_formation = True

            return (5*side_mod, 0)

    def is_ball_measurable(self):
        """
        Are the distance and dir of the ball accessible.
        """

        return (self.wm.ball.distance is not None \
                and self.wm.ball.direction is not None)

    def get_random_pos(self):
            # kick the ball to random position
            kick_to_pos_x = np.random.uniform(low=-5, high=0.)
            kick_to_pos_y = np.random.uniform(low=-20, high=20)
            
            return (kick_to_pos_x, kick_to_pos_y)

    def move_to_kick_ball(self, kick_power, kick_dir, vel, role="agent"):
        ## the player should move to the ball
        ## with a constant speed in order to 
        ## perform meaningful dribbling and kicking
        ## slow down the player as it approaches
        if role == "agent":
            player_vel = vel
        else:
            player_vel = vel-30

        speed_dir = 0.
        if self.wm.ball is not None:
            if self.is_ball_measurable():
                speed_dir = self.wm.ball.direction
                if self.wm.ball.distance <= 5:
                    if role == "agent":
                        player_vel = 60
                    else:
                        player_vel = 30

        self.wm.ah.turn(speed_dir)
        self.wm.align_neck_with_body()
        self.wm.ah.dash(player_vel, speed_dir)
        self.last_kick_dir = speed_dir

        if self.wm.is_ball_kickable() and role == "agent":
            self.wm.ah.kick(kick_power, kick_dir)

    def think(self, action, done, rewards, role="trainer"):
        """
        Performs a single step of thinking for our agent. Gets called on every
        iteration of our think loop.
        """
        reset_flag = None
        if self.send_commands:
            self.send_commands = False
            self.wm.ah.send_commands()
        
        if self.should_think_on_data:
            
            self.should_think_on_data = False

            is_nearer_to_goal = False
            # get the ball's distance to enemy goal:
            ## get enemy goal id
            if self.wm.side == "l":
                enemy_goal_id = "r"
            else:
                enemy_goal_id = "l"
            
            goal_to_player = []
            if self.wm.ball is not None:
                for goal in self.wm.goals:
                    if goal.goal_id == enemy_goal_id:
                        goal_to_player.append(goal.distance)
                        goal_to_player.append(goal.direction)
            
            ###################### SENDING COMMANDS ########################
            #################################################################
            
            # the agent should find the ball if ball not in sight
            if self.wm.ball is None or self.wm.ball.direction is None:
                self.wm.ah.turn(30)

            elif self.wm.ball is not None and not self.wm.is_before_kick_off():
                if not self.is_in_right_side:
                    while self.take_the_ball_to(point=(0., 0.)):
                        pass
                else:
                    kick_power = action[2]
                    kick_dir = action[3]
                    self.move_to_kick_ball(kick_power=kick_power, kick_dir=kick_dir, vel=70., role=role)
            
            ############# GET UPDATES ###########
            ##########################################
                
            # get current position tuple: ( , )
            curr_pos = self.wm.abs_coords
            
            # always in the enemy side of the field
            if curr_pos[0] < -5 and role == "agent":
                self.is_in_right_side = False
            else:
                self.is_in_right_side = True

            ## get enemy goal distance and dir relative to player
            ## [dist, dir]
            for goal in self.wm.goals:
                if goal.goal_id == enemy_goal_id:
                    goal_to_player.append(goal.distance)
                    goal_to_player.append(goal.direction)
       
            ## get dist and dir of the ball to player
            ## [dist, dir]
            ## UPDATE ONLY BALL IS NOT NONE
            if self.wm.ball is not None:
                ball_to_player = []
                ball_to_player.append(self.wm.ball.distance)
                ball_to_player.append(self.wm.ball.direction)
            
                ############## STATES ################
                # update the state of the player and ball
                # used to determine if an episode has ended
                self.state = np.array([curr_pos[0], curr_pos[1]])
                
                # determine if the ball is in range
                # by checking if play_mode is drop ball
                ball_out_range = (self.wm.is_kick_in() or
                                  self.wm.is_goal_kick() or 
                                  self.wm.is_dead_ball_them())

                # determine if the player comes nearer to the goal
                if len(goal_to_player) == 4:
                    is_nearer_to_goal = (goal_to_player[2] - goal_to_player[0] > 0)
                
                # determine if player is in shooting range
                is_in_shoot_range = False
                goal_dist = 50
                for goal in self.wm.goals:
                    if goal.goal_id == enemy_goal_id:
                        goal_dist = goal.distance
                
                is_in_shoot_range = (goal_dist < 20.)

                # # stop if player stamina is smaller than 1
                # stamina = self.wm.get_stamina()

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
                if role is not "trainer":
                    if self.wm.is_scored():
                        rewards = 10
                        # self.wm.kick_count = 0
                        done = True
                        reset_flag = 'goal'

                    elif ball_out_range:
                        rewards -= 5
                        reset_flag = 'not_in_range'
                        # self.wm.kick_count = 0
                        done = True

                    elif is_nearer_to_goal:
                        rewards += 1

                    elif is_in_shoot_range:
                        rewards += 2
                        
                    elif not self.is_in_right_side:
                        rewards -= 1
                        
                    # elif self.wm.kick_count > 10:
                    #     rewards -= 1
                    #     self.wm.kick_count = 0
                    #     done = True
                    
                    else:
                        rewards -= 0.5
        
        # if self.wm.kick_count >= 10:
        #     self.wm.kick_count = 0
        
        self.rewards = rewards
        self.done = done
        self.kick_off = False

        return reset_flag, self.rewards, self.done

    def take_the_ball_to(self, point):
        """
        Let the agent dribble the ball to point.
        """
        check = False
        ball_pos = None
        while not check:
            if self.send_commands:
                self.send_commands = False
                self.wm.ah.send_commands()

            # only think if new data has arrived
            if self.should_think_on_data:
                self.should_think_on_data = False
                
                if self.wm.ball.distance is None:
                    self.wm.ah.turn(20)
                    
                else:
                    self.move_to_kick_ball(kick_power=30, kick_dir=0., vel=40)
                    ball_pos = self.wm.get_object_absolute_coords(self.wm.ball)
                
                if ball_pos is not None:
                    dist_diff = self.wm.euclidean_distance(ball_pos, point)
                    if dist_diff <= 10:
                        check = True
                        
        return True

    def pass_to_player(self, side, unum):
        """
        Pass the ball to a player pos 
        """
        check = False
        while not check:
            if self.send_commands:
                self.send_commands = False
                self.wm.ah.send_commands()

            # only think if new data has arrived
            if self.should_think_on_data:
                self.should_think_on_data = False

                # if no player in sight, turn body
                if self.wm.get_player_coords(side, unnum=unum) is None:
                    self.wm.ah.turn(30)
                    
                else:
                    # get the player pos in sight
                    pass_to = self.wm.get_player_coords(side, unnum=unum)
                    if pass_to is not None:
                        self.wm.kick_to(point=pass_to, extra_power=-0.3)

                    # self.wm.ah.kick(20, 0.)
                # curr_pos = self.wm.abs_coords
                # if self.wm.is_ball_kickable():
                #     self.wm.kick_to(point=(curr_pos[0], curr_pos[1] + 10))
                # else:
                #     self.wm.ah.dash(50, self.wm.ball.direction)

                if self.wm.play_mode == world_model.WorldModel.PlayModes.PLAY_ON:
                    check = True

        return True

    def follow_the_ball(self, is_kick=False):
        self.trainer_count += 1
        # turn the body of the opponent to ball
        if self.send_commands:
            self.send_commands = False
            self.wm.ah.send_commands()

        # only think if new data has arrived
        if self.should_think_on_data:
            self.should_think_on_data = False
            
            if self.wm.ball.distance is None or self.wm.ball.direction is None:
                self.wm.ah.turn(30)
                self.wm.align_neck_with_body()

            elif self.wm.ball is not None:
                # print("follow the ball " + str(self.trainer_count))
                if self.is_ball_measurable():
                    if self.wm.ball.distance >= 10:
                        self.wm.turn_body_to_object(self.wm.ball)
                        self.wm.ah.dash(60, self.wm.ball.direction)
                    else:
                        if not is_kick:
                            self.wm.turn_body_to_object(self.wm.ball)
                        elif is_kick and not self.wm.is_ball_kickable()\
                            and self.wm.play_mode != self.wm.PlayModes.IS_GOAL:
                            self.wm.ah.dash(50, self.wm.ball.direction)
                        elif is_kick and self.wm.is_ball_kickable():
                                if self.pass_to_player(side='r', unum=2):
                                    return True
                                
                else:
                    # print("follow the player " + str(self.trainer_count))
                    player_coord = self.wm.get_player_coords(side='l', unnum=1)
                    if player_coord is not None:
                        self.wm.turn_body_to_point(player_coord)
                        self.wm.ah.dash(50, 0.)
                    else:
                        self.wm.ah.dash(50, 0.)

        return False

    def trainer(self):
        if self.reset_flag == 'goal':
            # move the opponent to ball
            # ONLY AGENT CAN SEE THE BALL
            flag = False
            check = False
            while not check:
                flag = self.follow_the_ball(is_kick=True)
                if flag and self.wm.play_mode == self.wm.PlayModes.PLAY_ON:
                    check = True
                    self.done = False
                    self.rewards = 0.
                    return

        elif self.reset_flag == 'not_in_range':
            if self.wm.ball is not None:
                if self.is_ball_measurable():
                    flag = False
                    check = False
                    while not check:
                        flag = self.follow_the_ball(is_kick=True)
                        if flag and self.wm.play_mode == self.wm.PlayModes.PLAY_ON:
                            check = True
                            self.done = False
                            self.rewards = 0.
                            return


if __name__ == "__main__":
    import sys
    import multiprocessing as mp

    def spawn_agent(team_name, unnum=1):
        """
        Used to run an agent in a seperate physical process.
        """

        a = Agent()
        a.connect("localhost", 6000, team_name, unnum, side=world_model.WorldModel.SIDE_L)
        a.play()

        # we wait until we're killed
        while 1:
            # we sleep for a good while since we can only exit if terminated.
            time.sleep(1)

    teamname = "my_team"
    spawn_agent(teamname)

    print("Spawned agents.")
    print()
    print("Playing soccer...")

