from qulacs import ParametricQuantumCircuit, QuantumState, DensityMatrix
from qulacs.gate import CNOT
from qulacs.gate import *

import numpy as np
from typing import List, Callable, Optional, Dict

from scipy.optimize import OptimizeResult
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit_qulacs import QulacsProvider
from qiskit_aer import Aer
# backend = QulacsProvider().get_backend('qulacs_simulator')
simulator = AerSimulator(method='matrix_product_state')



class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = QuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.cx([ctrl[r].item()], [targ[r].item()])
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        self.ansatz.rx(thetas[0][rot_qubit].item(), rot_qubit.item())
                    elif r == 1:
                        self.ansatz.ry(thetas[1][rot_qubit].item(), rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[2][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        return self.ansatz


def get_energy_qulacs(angles, observable,circuit, n_qubits, n_shots,
                      phys_noise = False,
                      which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    phys_noise            [bool]       : Whether quantum error channels are available (DM simulation) 
    
    Output:
    expval [float] : expectation value 
    
    """
    
    # print(circuit)

    no = 0
    for i in circuit:
        gate_detail = list(i)[0]
        if gate_detail.name in ['rx', 'ry', 'rz']:
            list(i)[0].params = [angles[no]]
            no+=1
    
    # result = backend.run(circuit).result()
    statevector = Statevector(circuit)
    # statevector = result.get_statevector()
    state = np.asmatrix(statevector)
    energy = (state @ observable) @ state.getH()

    return float(energy.real)

def get_exp_val(n_qubits,circuit,op, phys_noise = False, err_mitig = 0):

    # print('---------------')
    # print(circuit)
    # print('---------------')
    
    statevector = Statevector(circuit)
    # circuit.save_statevector(label='my_sv')
    # circuit.save_matrix_product_state(label='my_mps')
    # tcirc = transpile(circuit, simulator)
    # result = simulator.run(tcirc).result()
    # data = result.data(0)['my_mps']
    # print(data[0])
    # print(data[1])
    # print(result.data(0))
    # exit()
    state = np.asmatrix(statevector)
    energy = (state @ op) @ state.getH()
    # print(energy)
    # exit()

    return float(energy.real)



if __name__ == "__main__":
    pass


















