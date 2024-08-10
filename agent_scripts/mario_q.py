import numpy as np
import gym_super_mario_bros
import time
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
from gym import Wrapper
from gym.wrappers import ResizeObservation, FrameStack
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import psutil

ENV_NAME = 'SuperMarioBros-1-1-v0'
SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 50000
BUFFER_SIZE=150000


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        # create a reward accumulator
        reward_accum = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, info = self.env.step(action)
            # add reward to accumulator
            reward_accum += reward
            if done:
                break
        return next_state, reward_accum, done, info
    
    '''
    def reset(self):
        state = self.env.reset()
        print(state)
        # Assuming we want to stack 'skip' frames
        state = [state] * self.skip
        state = np.stack(state, axis=0)
        return state
    '''

def apply_wrappers(env):
    env = SkipFrame(env, skip=4) # skip every four frames
    env = ResizeObservation(env, shape=84) # reduce size of frame image
    env = GrayScaleObservation(env) # create grayscale images
    env = FrameStack(env, num_stack=4, lz4_compress=True) # stack frames (4 skipped)
    return env



class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Conolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # use built-in method to get the dimensional input size for initial linear layer
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected linear layers
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions) # determine best action to predict
        )

        # call the freeze method if frozen
        # to make sure no parameters are updated if frozen
        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # try to use the GPU if possible
        self.to(self.device)

    # method to handle forward pass 
    def forward(self, x):
        # pass the input tensor through the neural network layers
        return self.network(x)

    # get the number of neurons for our linear layers
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    # method to make sure gradients are not calculated if frozen
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False



class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=BUFFER_SIZE, 
                 batch_size=32, 
                 sync_network_rate=10000
                 ):
        
        self.num_actions = num_actions # use the appropriate number of actions (SIMPLE_MOVEMENT dict has 7)
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss() # loss function

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        self.log_memory_usage()

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

    def choose_action(self, observation):
        # create the potential to choose a random action
        # this will include some value of randomness to increase exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        observation = (
            torch.tensor(np.array(observation), dtype=torch.float32) # speed up processing by using tensors instead of numpy arrays
            .unsqueeze(0) # add dimension of batch size to first index of tensor
            .to(self.online_network.device) # move to the correct device (GPU or CPU)
        )
        # return the action with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    # compute the value of epsilon to diminish rewards for later actions
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    # put tensors in a dict and add to buffer
    def store_in_memory(self, state, action, reward, next_state, done):
        # Create TensorDict with correct shapes and types
        data = TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done)
        }, batch_size=[])
        self.replay_buffer.add(data)
    
    # copy weights of online network to target network if enough steps have passed
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    # save current model (in case something goes wrong)
    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    # load model
    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))


    def learn(self):
        # if not enough experiences, return and keep going
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # copy weights to target network
        self.sync_networks()
        
        # clear gradients
        self.optimizer.zero_grad()

        # sample the replay buffer and store the results
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        states = samples['state']
        actions = samples['action']
        rewards = samples['reward']
        next_states = samples['next_state']
        dones = samples['done']

        # get the predicted values from our neural network with the appropriate batch size
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()




env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)


env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

def print_progress(cur,end, bar_length=40):
    progress = cur / end
    block = int(round(bar_length * progress))
    text = f"\rCurrently processing episode {cur}/{end} [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
    print(text, end='', flush=True)

for i in range(NUM_OF_EPISODES):
    print_progress(i+1,NUM_OF_EPISODES)
    done = False
    state = env.reset()
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, info  = env.step(a)
        
        agent.store_in_memory(state, a, reward, new_state, done)
        agent.learn()

        state = new_state

        if (i + 1) % SAVE_INTERVAL == 0:
            agent.save_model(os.path.join("trained_agents","q_learning",str(i + 1) + "_q_agent.pt"))

agent.save_model(os.path.join("trained_agents","q_learning","final_q_agent.pt"))

env.close()