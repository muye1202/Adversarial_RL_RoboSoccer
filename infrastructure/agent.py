#!/usr/bin/env python

import threading
import time
from . import sock
from . import sp_exceptions
from . import handler
from . import world_model

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
        self.__should_think_on_data = False

        # whether we should send commands
        self.__send_commands = False
        
        # whether updates from server is obtained
        self.recv_message = False

    def connect(self, host, port, teamname, unnum, version=17):
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
        init_address = self.__sock.address
        init_msg = "(init %s (version %d))"
        init_msg = (init_msg % (teamname, version))
        self.__sock.send(init_msg)

        # wait until the socket receives a response from the server and gets its
        # assigned port.
        while self.__sock.address == init_address:
            time.sleep(0.0001)

        ##### NOT NEEDED DURING TRAINING ###
        # # create our thinking thread.  this will perform the actions necessary
        # # to play a game of robo-soccer.
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
        # # tell the thread that it should be running, then start it
        # self.__thinking = True
        # self.__should_think_on_data = True
        # # self.__think_thread.start() TODO: __think__thread is not starting
        # self.__think_loop()

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

        print("entered message loop")
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
                self.__send_commands = True
                self.recv_message = True

            # flag new data as needing the think loop's attention
            self.__should_think_on_data = True
            
    def received_update(self):
        
        return self.recv_message

    def __think_loop(self):
        """
        Performs world model analysis and sends appropriate commands to the
        server to allow the agent to participate in the current game.

        Like the message loop, this SHOULD NOT be called externally.  Use the
        play method to start play, and the disconnect method to end it.
        """
        print("entered think loop")
        while self.__thinking:
            # tell the ActionHandler to send its enqueued messages if it is time
            if self.__send_commands:
                self.__send_commands = False
                self.wm.ah.send_commands()

            # only think if new data has arrived
            if self.__should_think_on_data:
                # flag that data has been processed.  this shouldn't be a race
                # condition, since the only change would be to make it True
                # before changing it to False again, and we're already going to
                # process data, so it doesn't make any difference.
                self.__should_think_on_data = False

                # performs the actions necessary for the agent to play soccer
                self.think()
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

    def think(self):
        """
        Performs a single step of thinking for our agent. Gets called on every
        iteration of our think loop.
        """
        # take places on the field by uniform number
        if not self.in_kick_off_formation:
            print("enter team formation stage")

            # used to flip x coords for other side
            side_mod = 1
            if self.wm.side == world_model.WorldModel.SIDE_R:
                side_mod = -1
            
            self.wm.teleport_to_point((5 * side_mod, 0))
            self.in_kick_off_formation = True

            return

        # determine the enemy goal position
        goal_pos = None
        if self.wm.side == world_model.WorldModel.SIDE_R:
            goal_pos = (-55, 0)
        else:
            goal_pos = (55, 0)

        # kick off!
        if self.wm.is_before_kick_off():
            print("kick off")
            # player takes the kick off
            if self.wm.is_ball_kickable():
                # kick with 100% extra effort at enemy goal
                self.wm.kick_to(goal_pos, 1.0)
            else:
                # move towards ball
                if self.wm.ball is not None:
                    if (self.wm.ball.direction is not None and
                            -7 <= self.wm.ball.direction <= 7):
                        self.wm.ah.dash(20, 0.)
                    else:
                        self.wm.turn_body_to_point((0, 0))

                # turn to ball if we can see it, else face the enemy goal
                if self.wm.ball is not None:
                    self.wm.turn_neck_to_object(self.wm.ball)

                return

        # attack!
        else:
            # find the ball
            if self.wm.ball is None or self.wm.ball.direction is None:
                self.wm.ah.turn(30)

                return

            # kick it at the enemy goal
            if self.wm.is_ball_kickable():
                self.wm.kick_to(goal_pos, 1.0)
                return
            else:
                # move towards ball
                if -7 <= self.wm.ball.direction <= 7:
                    self.wm.ah.dash(20, 0.)
                else:
                    # face ball
                    self.wm.ah.turn(self.wm.ball.direction / 2)

                return


if __name__ == "__main__":
    import sys
    import multiprocessing as mp

    # # enforce corrent number of arguments, print help otherwise
    # if len(sys.argv) < 3:
    #     print("args: ./agent.py <team_name> <num_players>")
    #     sys.exit()

    def spawn_agent(team_name, unnum=1):
        """
        Used to run an agent in a seperate physical process.
        """

        a = Agent()
        a.connect("localhost", 6000, team_name, unnum)
        a.play()

        # we wait until we're killed
        while 1:
            # we sleep for a good while since we can only exit if terminated.
            time.sleep(1)

    # spawn all agents as seperate processes for maximum processing efficiency
    # agentthreads = []
    # for agent in range(1):
    #     print("  Spawning agent %d..." % agent)

    #     teamname = "my_team"
    #     at = mp.Process(target=spawn_agent, args=teamname)
    #     at.daemon = True
    #     at.start()

    #     agentthreads.append(at)
    teamname = "my_team"
    spawn_agent(teamname)
    # at = mp.Process(target=spawn_agent, args=str(teamname))
    # at.daemon = True
    # at.start()

    print("Spawned agents.")
    print()
    print("Playing soccer...")

    # # wait until killed to terminate agent processes
    # try:
    #     while 1:
    #         time.sleep(0.05)
    # except KeyboardInterrupt:
    #     print()
    #     print("Killing agent threads...")

    #     # terminate all agent processes
    #     count = 0
    #     for at in agentthreads:
    #         print("  Terminating agent %d..." % count)
    #         at.terminate()
    #         count += 1
    #     print("Killed %d agent threads." % (count - 1))

    #     print()
    #     print("Exiting.")
    #     sys.exit()

