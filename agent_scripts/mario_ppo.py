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

# Create the Super Mario Bros. environment
env = gym.make('SuperMarioBros-v0')
steps = env._max_episode_steps

# Set the Joypad wrapper
env = JoypadSpace(env.env, SIMPLE_MOVEMENT)
# Overwrite the old reset to accept seeds and options args
env.reset = gymnasium_reset.__get__(env, JoypadSpace)
env = StepAPICompatibility(env, output_truncation_bool=True)

model = PPO(
    'CnnPolicy',      # Use a convolutional neural network
    env,              # environment
    verbose=1,        # print diagnostics
    learning_rate=1e-4,  # controls how much to adjust the model with each step
    n_steps=128,      # affects the frequency of updates
    batch_size=64,    # number of samples per gradient update
    n_epochs=4,       # Number of epochs
    clip_range=0.2,   # helps in limiting updates for stable training
    ent_coef=0.03     # use entropy to encourage exploration
)

# Define evaluation and checkpoint callbacks
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./trained_agents/logs/', eval_freq=500, deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./trained_agents/', name_prefix='ppo_mario')

model.learn(total_timesteps=1500, callback=[eval_callback, checkpoint_callback])

# model = PPO.load("ppo_mario.zip")

model.save("./trained_agents/ppo_mario")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
env.close()
print(f"Mean reward: {mean_reward} ± {std_reward}")
