import sys
import os
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
from infrastructure import agent
from infrastructure.sim_process import run_sim

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
model.learn(total_timesteps=1e4, log_interval=1)

# # test environment
# episodes = 5
# for episode in range(episodes):
#     state = robo_soccer.reset()
#     done = False
#     score = 0
    
#     while not done:
#         action = robo_soccer.action_space.sample()
#         _, rewards, done, info = robo_soccer.step(action)
#         score += rewards
    
#     print('Episode:{} Score:{}'.format(episode, score))
