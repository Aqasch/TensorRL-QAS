import torch
from sys import stdout
import scipy
import os
import numpy as np
import copy

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from itertools import product

# Import your curriculum module
import curricula


class CircuitEnv():

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.max_episode_length = conf['env'].get('max_episode_length', 130)
        self.tolerance = conf['env'].get('tolerance', 0.99) 
        self.ham_type = 'quantum_compile'
        
        self.curriculum_dict = {}
        self.curriculum_dict[self.ham_type] = curricula.__dict__[conf['env']['curriculum_type']](
            conf['env'], target_energy=0  # 0 distance = perfect fidelity
        )
        
        self.mode = conf['env'].get('mode', 'paper')  # 'paper' or 'matrix'
        self.basis_type = 'rotations'
        self.reward_type = conf['env'].get('reward_type', 'dense')
        self.device = device

        # Initialize action space
        self._init_action_space()
        
        # Target unitary (from paper example, Equation 1)
        self.target_unitary = np.array([
            [0.76749896 - 0.43959894j, -0.09607122 + 0.45658344j],
            [0.09607122 + 0.45658344j,  0.76749896 + 0.43959894j]
        ], dtype=np.complex128)
        
        # State size for observation
        dim = 2 ** self.num_qubits
        self.state_size = 2 * dim * dim  # Flattened real/imag parts
        
        # Tracking
        self.step_counter = -1
        self.prev_fidelity = None
        self.action_list = []
        self.current_unitary = None
        
        stdout.flush()

    def _init_action_space(self):
        """Initialize action space for rotation gates"""
        # 6 gates: Rx(±π/8), Ry(±π/8), Rz(±π/8)
        self.action_size = 6
        self.gate_angle = np.pi / 8  # Use π/8 as in paper
        
        self.action_to_gate = {
            0: ('rx',  self.gate_angle),   # Rx(π/8)
            1: ('rx', -self.gate_angle),   # Rx(-π/8)
            2: ('ry',  self.gate_angle),   # Ry(π/8)
            3: ('ry', -self.gate_angle),   # Ry(-π/8)
            4: ('rz',  self.gate_angle),   # Rz(π/8)
            5: ('rz', -self.gate_angle),   # Rz(-π/8)
        }
        
        self.gate_names = [
            'Rx(π/8)', 'Rx(-π/8)', 
            'Ry(π/8)', 'Ry(-π/8)', 
            'Rz(π/8)', 'Rz(-π/8)'
        ]

    def _matrix_to_state_vector(self, matrix):
        """Convert complex matrix to real-valued state vector"""
        flat = matrix.flatten()
        state = np.concatenate([flat.real, flat.imag])
        return state.astype(np.float32)
    
    def _get_observation(self):
        """Get observation based on mode"""
        if self.mode == 'matrix':
            # Mode 1: Direct current unitary
            obs_matrix = self.current_unitary
        else:  # paper mode
            # Mode 2: O_n = U_target^† @ U_current
            obs_matrix = self.target_unitary.conj().T @ self.current_unitary
        
        return self._matrix_to_state_vector(obs_matrix)

    def step(self, action, train_flag=True):
        """
        Take a step by applying a gate from the action space
        """
        self.step_counter += 1
        self.action_list.append(action)
        
        # Update current unitary by applying the gate
        gate_type, angle = self.action_to_gate[action]
        gate_matrix = self._get_gate_matrix(gate_type, angle)
        self.current_unitary = gate_matrix @ self.current_unitary
        
        # Get new observation
        next_state = self._get_observation()
        
        # Compute fidelity
        fidelity = self.average_gate_fidelity()
        
        # Track best fidelity (HIGHER is better)
        if train_flag and fidelity > self.curriculum.lowest_energy:
            self.curriculum.lowest_energy = copy.copy(fidelity)
        
        # Check termination
        fidelity_achieved = fidelity >= self.tolerance
        max_steps_reached = self.step_counter >= (self.max_episode_length - 1)
        done = fidelity_achieved or max_steps_reached
        
        # Compute reward
        self.error = 1.0 - fidelity
        reward = self.compute_reward(fidelity, fidelity_achieved)
        
        # Update curriculum on episode end
        if done:
            self.curriculum.update_threshold(energy_done=fidelity_achieved)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[self.current_prob] = copy.deepcopy(self.curriculum)
        
        # Store for next step
        self.prev_fidelity = fidelity
        
        # print(f"Step {self.step_counter}: action={self.gate_names[action]}, "
        #       f"fidelity={fidelity:.6f}, error={self.error:.6f}, reward={reward:.4f}")
        
        return (torch.tensor(next_state, dtype=torch.float32, device=self.device),
                torch.tensor(reward, dtype=torch.float32, device=self.device),
                done)

    def reset(self):
        """Reset environment for new episode"""
        # Reset tracking
        self.step_counter = -1
        self.action_list = []  # IMPORTANT: Clear action list!
        
        # Start from identity
        dim = 2 ** self.num_qubits
        self.current_unitary = np.eye(dim, dtype=np.complex128)
        
        # Reset curriculum
        self.current_prob = self.ham_type
        self.curriculum = copy.deepcopy(self.curriculum_dict[self.current_prob])
        self.done_threshold = self.curriculum.get_current_threshold()
        
        # Initial fidelity
        self.prev_fidelity = self.average_gate_fidelity()
        
        # Get initial observation
        state = self._get_observation()
        
        # print(f'\n=== NEW EPISODE ===')
        # print(f'Initial fidelity: {self.prev_fidelity:.6f}')
        
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _get_gate_matrix(self, gate_type, angle):
        """Get 2x2 gate matrix for single-qubit rotation"""
        if gate_type == 'rx':
            # Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
        
        elif gate_type == 'ry':
            # Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        
        elif gate_type == 'rz':
            # Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
            return np.array([
                [np.exp(-1j * angle / 2), 0],
                [0, np.exp(1j * angle / 2)]
            ], dtype=np.complex128)
        
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    def make_circuit(self):
        """Construct Qiskit circuit from action list (for visualization)"""
        circuit = QuantumCircuit(self.num_qubits)
        
        for action in self.action_list:
            gate_type, angle = self.action_to_gate[action]
            
            if gate_type == 'rx':
                circuit.rx(angle, 0)
            elif gate_type == 'ry':
                circuit.ry(angle, 0)
            elif gate_type == 'rz':
                circuit.rz(angle, 0)
        
        return circuit

    def average_gate_fidelity(self):
        """
        Compute average gate fidelity between target and current unitaries
        AGF = (|Tr(U_target^† @ U_current)|^2 + d) / (d(d+1))
        """
        d = self.current_unitary.shape[0]
        trace = np.trace(self.target_unitary.conj().T @ self.current_unitary)
        fidelity = (np.abs(trace) ** 2 + d) / (d * (d + 1))
        return float(fidelity)
    
    def compute_reward(self, fidelity, fidelity_achieved):
        """Compute reward based on reward type (from paper)"""
        if self.reward_type == 'dense':
            # Dense reward: Equation (2) from paper
            # r = (L - n + 1) if solved, else -distance/L
            L = self.max_episode_length
            n = self.step_counter
            
            if fidelity_achieved:
                reward = (L - n + 1) / L  # Normalize
            else:
                distance = 1.0 - fidelity
                reward = -distance
                
        elif self.reward_type == 'sparse':
            # Sparse reward: Equation (4) from paper
            L = self.max_episode_length
            if fidelity_achieved:
                reward = 0.0
            else:
                reward = -1.0 / L
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        return reward
    
    def get_episode_info(self):
        """Get information about current episode"""
        fidelity = self.average_gate_fidelity()
        circuit = self.make_circuit()
        
        return {
            'fidelity': fidelity,
            'steps': self.step_counter + 1,
            'sequence_length': len(self.action_list),
            'action_sequence': self.action_list.copy(),
            'gate_names': [self.gate_names[a] for a in self.action_list],
            'solved': fidelity >= self.tolerance,
            'circuit': circuit,
            'error': 1.0 - fidelity
        }


if __name__ == "__main__":
    # Example usage
    config = {
        'env': {
            'num_qubits': 1,
            'max_episode_length': 130,
            'tolerance': 0.99,
            'reward_type': 'dense',
            'mode': 'paper',  # Use paper's observation method
            'curriculum_type': 'VanillaCurriculum',
            'thresholds': [0.99],
            'switch_episodes': [10000],
            'accept_err': 0.01
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = CircuitEnv(config, device)
    
    print(f"Environment initialized:")
    print(f"  State size: {env.state_size}")
    print(f"  Action size: {env.action_size}")
    print(f"  Gates: {env.gate_names}")
    print(f"  Target unitary:\n{env.target_unitary}")
    
    # Test episode
    state = env.reset()
    print(f"\nState shape: {state.shape}")
    
    done = False
    step_count = 0
    while not done and step_count < 100:
        action = np.random.randint(0, env.action_size)
        next_state, reward, done = env.step(action)
        step_count += 1
    
    info = env.get_episode_info()
    print(f"\n=== Episode Summary ===")
    print(f"Steps: {info['steps']}")
    print(f"Final fidelity: {info['fidelity']:.6f}")
    print(f"Solved: {info['solved']}")
    print(f"Gate sequence: {' -> '.join(info['gate_names'])}")
