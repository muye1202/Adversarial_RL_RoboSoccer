import sys
import os
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
from infrastructure import agent

from robo_env import RoboPlayer
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
    
teamname = "my_team"
player = agent.Agent()
enemy = agent.Agent()

# see training progress in tensorboard
logdir = 'logs_training'

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initiate the Environment
robo_soccer = RoboPlayer(role='attacker', agent=player, opponent=enemy, host="localhost", port=6000, teamname=teamname)
# check_env(robo_soccer)

model = DDPG('MlpPolicy', env=robo_soccer, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=20000, log_interval=1)