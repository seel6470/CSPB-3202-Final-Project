# Test the trained agent

import numpy as np
import gym_super_mario_bros
import time
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from gym.wrappers import StepAPICompatibility, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

# Create the Super Mario Bros. environment
env = gym.make('SuperMarioBros-v0')
steps = env._max_episode_steps

# Set the Joypad wrapper
env = JoypadSpace(env.env, SIMPLE_MOVEMENT)
# Overwrite the old reset to accept seeds and options args
def gymnasium_reset(self, **kwargs):
    return self.env.reset(), {}
env.reset = gymnasium_reset.__get__(env, JoypadSpace)

env = StepAPICompatibility(env, output_truncation_bool=True)
env = CustStepReward(env)

env = SkipFrame(env, skip=4)
env = ResizeObservation(env, shape=84) # reduce size of frame image
env = GrayScaleObservation(env) # create grayscale images
env = FrameStack(env, num_stack=8, lz4_compress=True) # stack frames

# uncomment to load saved trained model
model = PPO.load(os.path.join("trained_agents","ppo","ppo_mario_25000_steps.zip"))

max_x = 0
max_world = 0
max_stage = 0

obs, info = env.reset()
for step in range(2000):
    action, info = model.predict(np.array(obs))
    action = action.item()       
    obs, reward, done, truncated, info = env.step(action)
    if info['world'] > max_world:
        max_world = info['world']
        max_x = 0
        max_stage = 1
    elif info['stage'] > max_stage:
        max_stage = info['stage']
        max_x = 0
    elif info['world'] == max_world and info['stage'] == max_stage and info['x_pos'] > max_x:
        max_x = info['x_pos']
    env.render()
    time.sleep(0.05)
    if done:
        obs, info = env.reset()

# Close the environment
env.close()
print(f"Max X: {max_x}")
print(f"Max World: {max_world}")
print(f"Max Stage: {max_stage}")