import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


# ========== Complex Neural Network Components ==========

class ComplexLinear(nn.Module):
    """Complex-valued linear layer"""
    def __init__(self, in_features, out_features, bias=True, dtype=torch.complex64):
        super().__init__()
        base_dtype = torch.float32
        # Initialize with magnitude ~ N(0, 1/sqrt(in_features)), random phase
        sigma = 1.0 / np.sqrt(in_features)
        mag = torch.randn(out_features, in_features, dtype=base_dtype) * sigma
        phase = 2 * np.pi * torch.rand(out_features, in_features, dtype=base_dtype)

        W_real_init = mag * torch.cos(phase)
        W_imag_init = mag * torch.sin(phase)

        self.W_real = nn.Parameter(W_real_init)
        self.W_imag = nn.Parameter(W_imag_init)

        if bias:
            self.b_real = nn.Parameter(torch.zeros(out_features, dtype=base_dtype))
            self.b_imag = nn.Parameter(torch.zeros(out_features, dtype=base_dtype))
        else:
            self.register_parameter("b_real", None)
            self.register_parameter("b_imag", None)
        self.dtype = dtype

    def forward(self, z):
        W = torch.complex(self.W_real, self.W_imag).to(self.dtype)
        b = 0
        if self.b_real is not None:
            b = torch.complex(self.b_real, self.b_imag).to(self.dtype)
        return z @ W.t() + b


class PhasorLayer(nn.Module):
    """Phasor layer with unit-modulus complex weights"""
    def __init__(self, in_features, out_features, dtype=torch.complex64):
        super().__init__()
        # Store angles θ, weights are exp(iθ)
        base_dtype = torch.float32
        self.theta = nn.Parameter(0.01 * torch.randn(out_features, in_features, dtype=base_dtype))
        self.dtype = dtype

    def forward(self, z):
        W = torch.exp(1j * self.theta.to(self.dtype))  # unit-modulus weights
        h = z @ W.t()
        return h


class PhasorDQN(nn.Module):
    """
    DQN using Phasor layers for complex-valued state representation
    Takes complex state, outputs real Q-values
    """
    def __init__(self, state_size, action_size, hidden_dim=128, dtype=torch.complex64):
        super().__init__()
        self.dtype = dtype
        self.state_size = state_size
        # state_size should be 2*dim*dim for flattened real/imag
        # We convert to complex dimension
        self.complex_dim = state_size // 2
        
        # Phasor layers
        self.phasor1 = PhasorLayer(self.complex_dim, hidden_dim, dtype=dtype)
        self.phasor2 = PhasorLayer(hidden_dim, hidden_dim, dtype=dtype)
        
        # Output layer: complex to real Q-values
        # We use ComplexLinear then take magnitude
        self.complex_out = ComplexLinear(hidden_dim, action_size, dtype=dtype)
        
    def forward(self, x):
        # x is real-valued (B, 2*dim*dim) from environment
        # Convert to complex (B, dim*dim)
        batch_size = x.shape[0]
        half = self.complex_dim
        x_real = x[:, :half]
        x_imag = x[:, half:]
        z = torch.complex(x_real, x_imag).to(self.dtype)
        
        # Phasor layers
        h1 = self.phasor1(z)
        h1 = h1 / (torch.abs(h1) + 1e-12)  # Normalize
        
        h2 = self.phasor2(h1)
        h2 = h2 / (torch.abs(h2) + 1e-12)  # Normalize
        
        # Output complex values
        out_complex = self.complex_out(h2)
        
        # Convert to real Q-values: use magnitude
        # Alternative: could use real part, or learned combination
        q_values = torch.abs(out_complex)
        
        return q_values


class MLPDQN(nn.Module):
    """
    DQN using real-valued MLP with separate real/imag processing
    Takes concatenated real/imag state, outputs real Q-values
    """
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        # state_size is already 2*dim*dim (real + imag)
        
        # MLP with LeakyReLU (following the pattern from the code)
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_size)
        )
        
        # Initialize with proper scaling
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=np.sqrt(1/layer.in_features))
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # x is already real-valued (B, 2*dim*dim)
        return self.net(x)


class StandardDQN(nn.Module):
    """
    Standard DQN with SELU activations (from paper)
    For 'paper' mode
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # Network layers (as per Table 1 in paper)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights as per paper (lecun, lecun, glorot)"""
        # LeCun initialization for hidden layers
        nn.init.normal_(self.fc1.weight, mean=0, std=np.sqrt(1/self.fc1.in_features))
        nn.init.normal_(self.fc2.weight, mean=0, std=np.sqrt(1/self.fc2.in_features))
        
        # Glorot (Xavier) initialization for output layer
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # Zero bias
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """Forward pass with SELU activations"""
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)  # Linear output for Q-values
        return x


# ========== Replay Buffer ==========

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ========== DQN Agent ==========

class DQNAgent:
    """
    DQN Agent that supports different network architectures:
    - 'paper': Standard DQN with SELU (for O_n representation)
    - 'matrix_phasor': PhasorDQN (for direct unitary matrix)
    - 'matrix_mlp': MLPDQN (for direct unitary matrix)
    """
    
    def __init__(self, state_size, action_size, config, device, network_type='paper'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.network_type = network_type
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.99976)
        self.learning_rate = config.get('learning_rate', 0.0005)
        self.batch_size = config.get('batch_size', 1000)
        self.memory_size = config.get('memory_size', 10000)
        self.target_update_freq = config.get('target_update_freq', 1000)
        
        # Create networks based on type
        if network_type == 'paper':
            self.policy_net = StandardDQN(state_size, action_size).to(device)
            self.target_net = StandardDQN(state_size, action_size).to(device)
        elif network_type == 'matrix_phasor':
            hidden_dim = config.get('hidden_dim', 128)
            self.policy_net = PhasorDQN(state_size, action_size, hidden_dim=hidden_dim).to(device)
            self.target_net = PhasorDQN(state_size, action_size, hidden_dim=hidden_dim).to(device)
        elif network_type == 'matrix_mlp':
            hidden_dim = config.get('hidden_dim', 128)
            self.policy_net = MLPDQN(state_size, action_size, hidden_dim=hidden_dim).to(device)
            self.target_net = MLPDQN(state_size, action_size, hidden_dim=hidden_dim).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(self.memory_size)
        
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state.unsqueeze(0))
                return q_values.argmax(1).item()
            
    def act(self, state, mask_actions=None):
        state = state.unsqueeze(0)
        epsilon = False

        # Build a mask: 1 for allowed, 0 for forbidden
        mask = torch.ones(self.action_size, dtype=torch.bool, device=state.device)
        if mask_actions is not None:
            for a in mask_actions:
                mask[a] = False

        # Epsilon-greedy: only choose among allowed actions
        allowed_indices = mask.nonzero(as_tuple=False).view(-1)
        if allowed_indices.numel() == 0:
            raise RuntimeError("No available actions to select!")  # All actions exhausted in this episode

        if torch.rand(1).item() <= self.epsilon:
            rand_ac = allowed_indices[torch.randint(len(allowed_indices), (1,))].item()
            epsilon = True
            return (rand_ac, epsilon)
        act_values = self.policy_net.forward(state)[0]

        # Mask out forbidden actions by setting their q-value to -inf
        act_values_masked = act_values.clone()
        act_values_masked[~mask] = float('-inf')
        action = torch.argmax(act_values_masked).item()
        return action, epsilon
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'network_type': self.network_type,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        print(f"Model loaded from {filepath}")


# ========== Training Functions (same as before) ==========

def train_dqn(env, agent, config):
    """Training loop for DQN agent - saves metrics at each STEP"""
    num_episodes = config.get('num_episodes', 50000)
    max_steps_per_episode = config.get('max_steps_per_episode', 130)
    print_every = config.get('print_every', 100)
    save_every = config.get('save_every', 1000)
    
    # Step-level metrics
    step_data = {
        'episode': [],
        'step_in_episode': [],
        'global_step': [],
        'action': [],
        'reward': [],
        'fidelity': [],
        'error': [],
        'loss': [],
        'epsilon': [],
        'done': []
    }
    
    # Episode-level summary
    episode_rewards = []
    episode_lengths = []
    episode_fidelities = []
    success_rate_window = deque(maxlen=100)
    
    global_step = 0
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Network type: {agent.network_type}")
    print(f"State size: {agent.state_size}, Action size: {agent.action_size}")
    print(f"Epsilon decay: {agent.epsilon_decay}, Batch size: {agent.batch_size}")
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        step_in_episode = 0
        
        while not done and step_in_episode < max_steps_per_episode:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done = env.step(action, train_flag=True)
            
            # Store transition
            agent.store_transition(state, action, reward.item(), next_state, done)
            
            # Train
            loss = agent.train_step()
            
            # Get metrics
            current_fidelity = env.average_gate_fidelity()
            current_error = 1.0 - current_fidelity
            
            # Store step data
            step_data['episode'].append(episode + 1)
            step_data['step_in_episode'].append(step_in_episode + 1)
            step_data['global_step'].append(global_step)
            step_data['action'].append(action)
            step_data['reward'].append(reward.item())
            step_data['fidelity'].append(current_fidelity)
            step_data['error'].append(current_error)
            step_data['loss'].append(loss if loss is not None else np.nan)
            step_data['epsilon'].append(agent.epsilon)
            step_data['done'].append(int(done))
            
            state = next_state
            episode_reward += reward.item()
            step_in_episode += 1
            global_step += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        agent.episodes_done += 1
        
        # Episode info
        info = env.get_episode_info()
        episode_rewards.append(episode_reward)
        episode_lengths.append(info['steps'])
        episode_fidelities.append(info['fidelity'])
        success_rate_window.append(1.0 if info['solved'] else 0.0)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_length = np.mean(episode_lengths[-print_every:])
            avg_fidelity = np.mean(episode_fidelities[-print_every:])
            success_rate = np.mean(success_rate_window)
            avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Avg Fidelity: {avg_fidelity:.6f}")
            print(f"  Success Rate (last 100): {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg Loss: {avg_loss:.6f}")
            print(f"  Total Steps: {global_step}")
        
        # Save periodically
        if (episode + 1) % save_every == 0:
            agent.save(f'checkpoints/dqn_episode_{episode + 1}.pth')
            save_step_metrics(step_data, episode + 1)
    
    print("\nTraining completed!")
    save_step_metrics(step_data, num_episodes, final=True)
    
    return episode_rewards, episode_lengths, episode_fidelities, step_data


def save_step_metrics(step_data, episode_num, final=False):
    """Save step-level metrics"""
    os.makedirs('results', exist_ok=True)
    
    df = pd.DataFrame(step_data)
    
    if final:
        csv_path = 'results/step_metrics_final.csv'
        print(f"\nSaving final step metrics to {csv_path}")
        print(f"Total steps recorded: {len(df)}")
    else:
        csv_path = f'results/step_metrics_episode_{episode_num}.csv'
    
    df.to_csv(csv_path, index=False)
    
    if final:
        np.savez('results/step_metrics_final.npz',
                 episode=np.array(step_data['episode']),
                 step_in_episode=np.array(step_data['step_in_episode']),
                 global_step=np.array(step_data['global_step']),
                 action=np.array(step_data['action']),
                 reward=np.array(step_data['reward']),
                 fidelity=np.array(step_data['fidelity']),
                 error=np.array(step_data['error']),
                 loss=np.array(step_data['loss']),
                 epsilon=np.array(step_data['epsilon']),
                 done=np.array(step_data['done']))
        print(f"Saved final step metrics to results/step_metrics_final.npz")


def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate trained agent"""
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    
    successes = 0
    fidelities = []
    lengths = []
    errors = []
    
    eval_step_data = {
        'episode': [],
        'step_in_episode': [],
        'action': [],
        'fidelity': [],
        'error': [],
        'done': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < 130:
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action, train_flag=False)
            
            current_fidelity = env.average_gate_fidelity()
            eval_step_data['episode'].append(episode + 1)
            eval_step_data['step_in_episode'].append(step + 1)
            eval_step_data['action'].append(action)
            eval_step_data['fidelity'].append(current_fidelity)
            eval_step_data['error'].append(1.0 - current_fidelity)
            eval_step_data['done'].append(int(done))
            
            state = next_state
            step += 1
        
        info = env.get_episode_info()
        if info['solved']:
            successes += 1
        fidelities.append(info['fidelity'])
        errors.append(info['error'])
        lengths.append(info['steps'])
        
        if episode < 5:
            print(f"\nEpisode {episode + 1}:")
            print(f"  Fidelity: {info['fidelity']:.6f}")
            print(f"  Error: {info['error']:.6f}")
            print(f"  Solved: {info['solved']}")
            print(f"  Length: {info['steps']}")
            print(f"  Gate sequence: {' -> '.join(info['gate_names'][:10])}")
            if len(info['gate_names']) > 10:
                print(f"    ... ({len(info['gate_names'])} gates total)")
    
    print(f"\n=== Evaluation Results ===")
    print(f"Success Rate: {successes/num_episodes:.2%} ({successes}/{num_episodes})")
    print(f"Average Fidelity: {np.mean(fidelities):.6f}")
    print(f"Average Error: {np.mean(errors):.6f}")
    print(f"Average Length: {np.mean(lengths):.1f}")
    
    eval_summary_df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'fidelity': fidelities,
        'error': errors,
        'num_actions': lengths,
        'solved': [int(f >= 0.99) for f in fidelities]
    })
    eval_summary_df.to_csv('results/evaluation_summary.csv', index=False)
    
    eval_step_df = pd.DataFrame(eval_step_data)
    eval_step_df.to_csv('results/evaluation_step_metrics.csv', index=False)
    
    return successes / num_episodes, fidelities, lengths


# ========== Main ==========

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    from environment_qulacs_compiling import CircuitEnv
    
    # Configuration
    env_mode = 'matrix'  # 'paper' or 'matrix'
    network_type = 'matrix_phasor'  # 'paper', 'matrix_phasor', or 'matrix_mlp'
    
    env_config = {
        'env': {
            'num_qubits': 1,
            'max_episode_length': 130,
            'tolerance': 0.99,
            'reward_type': 'dense',
            'mode': env_mode,  # 'paper' or 'matrix'
            'curriculum_type': 'VanillaCurriculum',
            'thresholds': [0.99],
            'switch_episodes': [10000],
            'accept_err': 0.01
        }
    }
    
    agent_config = {
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.99976,
        'learning_rate': 0.0005,
        'batch_size': 1000,
        'memory_size': 10000,
        'target_update_freq': 1000,
        'hidden_dim': 128,  # For Phasor/MLP networks
    }
    
    training_config = {
        'num_episodes': 200,
        'max_steps_per_episode': 130,
        'print_every': 5,
        'save_every': 50,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment and agent
    env = CircuitEnv(env_config, device)
    agent = DQNAgent(env.state_size, env.action_size, agent_config, device, network_type=network_type)
    
    print("\n=== Environment Info ===")
    print(f"Mode: {env_mode}")
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    
    print("\n=== Agent Info ===")
    print(f"Network type: {network_type}")
    print(f"Total parameters: {sum(p.numel() for p in agent.policy_net.parameters())}")
    
    # Train
    episode_rewards, episode_lengths, episode_fidelities, step_data = train_dqn(
        env, agent, training_config
    )
    
    # Evaluate
    success_rate, eval_fidelities, eval_lengths = evaluate_agent(env, agent, num_episodes=100)
    
    # Save
    agent.save('checkpoints/dqn_final.pth')
    
    print("\n=== All results saved ===")
