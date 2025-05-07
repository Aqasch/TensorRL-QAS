import torch.nn as nn
import torch.nn.functional as F
import random
import torch

import copy
from collections import namedtuple, deque
import numpy as np
from itertools import product

from .utils import dictionary_of_actions_SU4, dict_of_actions_revert_q


class DQN_su4(object):

    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        memory_size = conf['agent']['memory_size']
        
        self.final_gamma = conf['agent']['final_gamma']
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        neuron_list = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']
        self.prioritized_replay = int(conf['agent']['priotitized_replay'])
        
        
        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch =  conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch =  False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False

        self.action_size = action_size

        self.state_size = state_size if self.with_angles else state_size - self.num_layers*self.num_qubits*3

        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size

        self.translate = dictionary_of_actions_SU4(self.num_qubits)
        self.rev_translate = dictionary_of_actions_SU4(self.num_qubits)
        self.policy_net = self.unpack_network(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma,1/self.num_layers),2)]).to(device)   
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size)
        else:
            self.memory = ReplayMemory(memory_size)

        self.epsilon = 1.0

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0

   
        self.Transition = namedtuple('Transition',
                            ('state', 'action', 'reward',
                            'next_state','done'))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, ill_action):
        state = state.unsqueeze(0)
        epsilon = False
        
        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:
                rand_ac = torch.randint(self.action_size, (1,)).item()
            epsilon = True
            return (rand_ac, epsilon)
        act_values = self.policy_net.forward(state)
        act_values[0][ill_action] = float('-inf') 

        return torch.argmax(act_values[0]).item(), epsilon

    def replay(self, batch_size):
        if self.step_counter %self.update_target_net ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1
        
        if self.prioritized_replay:
            indices, transitions, weights = self.memory.sample(batch_size, frame_idx=self.step_counter)
            weights = weights.to(self.device)
        else:
            weights = 1
            transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))
        
        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)
        

        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1))
        """ Double DQN """        
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_actions = self.policy_net.forward(next_state_batch).max(1)[1].detach()
        next_state_values = next_state_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
        
       
    
        """ Compute the expected Q values """
        expected_state_action_values = (next_state_values * self.gamma) * (1-done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        # Compute TD-errors for priority updates
        td_errors = torch.abs(expected_state_action_values - state_action_values)
        # Update priorities in memory
        if self.prioritized_replay:
            self.memory.update_priorities(indices, td_errors)

        # Apply importance-sampling weights
        # loss = self.loss(state_action_values * weights, expected_state_action_values * weights)

        assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"
        cost = self.fit(state_action_values, expected_state_action_values, weights)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon,self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Problem with epsilons"
        return cost

    def fit(self, output, target_f, weights):
        self.optim.zero_grad()
        loss = self.loss(output*weights, target_f*weights)
        loss.backward()
        self.optim.step()
        return loss.item()

    def unpack_network(self, neuron_list, p):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list 
        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(p=p))
        layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        return nn.Sequential(*layer_list)


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = []
        self.position = 0


class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Prioritized Replay Memory with Annealing Importance Sampling.
        
        Parameters:
        - capacity: Max size of the memory buffer.
        - alpha: Priority level (0 = uniform, 1 = full prioritization).
        - beta_start: Initial importance-sampling weight.
        - beta_frames: Number of frames over which beta increases to 1.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.alpha = alpha  # How much prioritization is used (0 = no prioritization, 1 = full)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        # Define a namedtuple to store experiences
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        """Save a transition with maximum priority (new experiences should be prioritized)."""
        max_priority = self.priorities.max() if self.memory else 1.0  # Avoid 0 priority on first push
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, frame_idx):
        """Sample a batch of experiences, with priority and importance sampling."""
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Calculate probabilities with prioritization (add epsilon to avoid zero priority)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probability distribution
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Anneal beta over time (to 1 as per beta_frames)
        self.beta = self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, self.beta)  # Cap beta at 1.0
        
        # Compute importance-sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Convert weights to tensor for PyTorch
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Unpack experiences for training
        # batch = self.Transition(*zip(*samples))
        
        # Return batch along with indices and weights for updating priorities
        return indices, samples, weights

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        """Update priorities of sampled transitions based on new TD-errors."""
        td_errors = td_errors.detach().cpu().numpy()
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + epsilon

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)


if __name__ == '__main__':
    pass