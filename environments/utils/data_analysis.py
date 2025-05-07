import pickle
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from utils import dictionary_of_actions
from qiskit import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16

})

def dictionary_of_actions(num_qubits):
    dictionary = dict()
    i = 0
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def make_circuit_qiskit(action, qubits, circuit):
    ctrl = action[0]
    targ = (action[0] + action[1]) % qubits
    rot_qubit = action[2]
    rot_axis = action[3]
#     print(1)
    if ctrl < qubits:
        circuit.cx([ctrl], [targ])
    if rot_qubit < qubits:
        if rot_axis == 1:
            circuit.rx(0, rot_qubit) # TODO: make a function and take angles
        elif rot_axis == 2:
            circuit.ry(0, rot_qubit)
        elif rot_axis == 3:
            circuit.rz(0, rot_qubit)
    
    return circuit

#S LOADING THE HAMILTONIAN
seed_agent = 1
optimizer_list = ['cobyla', 'lbfgs']
optimizer = optimizer_list[0]

for TN_state in [0,1]:

    if TN_state:
        tn_state = 'w_TN_init'

    else:
        tn_state = 'wo_TN_init'

    action_lenght_list = []
    succ_ep_list = []
    for prio_replay in [0]:
        data = np.load(f'results/finalize/vanilla_{optimizer}_LiH6q2p2_{tn_state}_prio_replay{prio_replay}/summary_{seed_agent}.npy',allow_pickle=True)[()]
        episodes = len(data['train'].keys())
        err_list = []
        for ep in range(0, episodes):
            err = data['train'][ep]['errors'][-1]
            err_list.append(err)
            if err <= 0.0016:
                succ_ep_list.append(ep)
                action_lenght_list.append(len(data['train'][ep]['actions']))
        print(np.argmin(err_list), np.min(err_list))
        if len(action_lenght_list) != 0 :
            shortest_ep = succ_ep_list[action_lenght_list.index(min(action_lenght_list))]
        else:
            shortest_ep = np.argmin(err_list)
        print()
        dic_actions = dictionary_of_actions(6)
        for action in data['train'][shortest_ep]['actions']:
            print(dic_actions[action])
        
        circuit = QuantumCircuit(6)
        for a in data['train'][shortest_ep]['actions']:
            action = dictionary_of_actions(6)[a]
            final_circuit = make_circuit_qiskit(action, 6, circuit)
        print(final_circuit)
        err_list_best_yet = data['train'][shortest_ep]['errors']
        plt.semilogy(err_list_best_yet, '-o', label = f'TN init: {TN_state} (tensor in state)')
plt.xlabel('Number of gates')
plt.xscale('log')
plt.ylabel('Error in ground state estimation')
plt.axvline(x=67, color = 'r', linestyle = '--', label = f'Curriculum RL')
plt.semilogy([0.0016]*70, 'k-x', label = f'CA')
plt.legend(fontsize = 12, ncol = 2)
plt.tight_layout()
plt.savefig('TN_and_without_TN_results.png')
plt.show()



