import sys
import os
sys.path.insert(0, '/home/muyejia1202/Robot_Soccer_RL/nu_robo_agent')
from infrastructure import agent
import matplotlib.pyplot as plt

from robo_env import RoboPlayer
from infrastructure.world_model import WorldModel
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
    
teamname = "my_team"
player = agent.Agent()

# initiate the Environment
robo_soccer = RoboPlayer('attacker', player, "localhost", 6000, teamname, 1)

# set log path and initiate model
log_path = os.path.join('training', 'logs')
env = Monitor(robo_soccer, log_path)

model = DDPG('MlpPolicy', env=robo_soccer)
model.learn(total_timesteps=20000, log_interval=10)

# plot results
results_plotter.plot_results(log_path, 20000, results_plotter.X_TIMESTEPS, 'robo_attacker')