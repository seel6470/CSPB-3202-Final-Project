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
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        # create a reward accumulator
        reward_accum = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, truncated, info = self.env.step(action)
            # add reward to accumulator
            reward_accum += reward
            if done:
                break
        return next_state, reward_accum, done, truncated, info

# final custom reward calculation after failed training (see below for details)
class CustStepReward(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_y_pos = 0
        self.prev_x_pos = 0
        self.reward_mean = 0
        self.reward_var = 1
        self.num_rewards = 0
        self.succesive_A_presses = 0

    def step(self, action):
        done = False
        next_state, reward, done, truncated, info = self.env.step(action)
        # if rising, agent must have pressed A state before
        if self.prev_y_pos < info['y_pos']:
            self.succesive_A_presses += 1
            reward += 2 * self.succesive_A_presses # incentivize jumping higher if already jumping
        # agent is not rising
        else:
            # reset to 0
            # no reward (and no penalty)
            self.succesive_A_presses = 0
        self.prev_y_pos = info['y_pos']

        # incentivize moving right
        if self.prev_x_pos < info['x_pos']:
            reward += 3
        # penalize staying still or moving left
        else:
            reward -= 8
        self.prev_x_pos = info['x_pos']
        
        if info['life'] < 2:
            reward -= 50 # heavily penalize death and end the episode
            done = True
        if info['flag_get']:
            reward += 100 # heavily incentivize beating the level
        
        # Update reward statistics
        self.num_rewards += 1
        old_mean = self.reward_mean
        self.reward_mean += (reward - self.reward_mean) / self.num_rewards
        self.reward_var += (reward - old_mean) * (reward - self.reward_mean)
        
        # Normalize reward
        reward_std = np.sqrt(self.reward_var / self.num_rewards)
        normalized_reward = (reward - self.reward_mean) / (reward_std + 1e-8)
        return next_state, normalized_reward, done, truncated, info

# Create the Super Mario Bros. environment
env = gym.make('SuperMarioBros-v0')
steps = env._max_episode_steps
CUSTOM_ACTIONS = [
    ['right', 'A', 'B'],
    ['right','B']
]
# Set the Joypad wrapper
env = JoypadSpace(env.env, SIMPLE_MOVEMENT)
# Overwrite the old reset to accept seeds and options args
def gymnasium_reset(self, **kwargs):
    return self.env.reset(), {}
env.reset = gymnasium_reset.__get__(env, JoypadSpace)

env = StepAPICompatibility(env, output_truncation_bool=True)
#env = CustStepReward(env)

#env = SkipFrame(env, skip=4)
#env = ResizeObservation(env, shape=84) # reduce size of frame image
#env = GrayScaleObservation(env) # create grayscale images
#env = FrameStack(env, num_stack=8, lz4_compress=True) # stack frames

# uncomment to load saved trained model
model = PPO.load(os.path.join("..","trained_agents","ppo","best_model.zip"))

print("###################\n\n",model.policy)

max_x = 0
max_world = 0
max_stage = 0

obs, info = env.reset()
for step in range(10000):
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
    if done:
        obs, info = env.reset()

# Close the environment
env.close()
print(f"Max X: {max_x}")
print(f"Max World: {max_world}")
print(f"Max Stage: {max_stage}")