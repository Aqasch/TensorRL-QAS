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
backend = QulacsProvider().get_backend('qulacs_simulator')
# simulator = AerSimulator(method='matrix_product_state')
simulator = AerSimulator(method='matrix_product_state')




class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = QuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        
        for _, local_state in enumerate(state):
            
            one_qubit_pos = (local_state[3*self.n_qubits: 3*self.n_qubits+3] == 1).nonzero( as_tuple = True )
            # print(local_state)
            xx_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            xx_theta = local_state[3*self.n_qubits+3:3*self.n_qubits+3 + self.n_qubits]

            yy_pos = (local_state[self.n_qubits:2*self.n_qubits] == 1).nonzero( as_tuple = True )
            yy_theta = local_state[3*self.n_qubits+3 + self.n_qubits:3*self.n_qubits+3 + 2*self.n_qubits]


            zz_pos = (local_state[2*self.n_qubits:3*self.n_qubits] == 1).nonzero( as_tuple = True )
            zz_theta = local_state[3*self.n_qubits+3 + 2*self.n_qubits:3*self.n_qubits+3 + 3*self.n_qubits]
            
            targ_xx = xx_pos[0]
            ctrl_xx = xx_pos[1]
            targ_yy = yy_pos[0]
            ctrl_yy = yy_pos[1]
            targ_zz = zz_pos[0]
            ctrl_zz = zz_pos[1]


            if len(ctrl_xx) != 0:
                for r in range(len(ctrl_xx)):
                    self.ansatz.rxx(xx_theta[targ_xx[r].item()][ctrl_xx[r].item()].item(), ctrl_xx[r].item(), targ_xx[r].item())
            if len(ctrl_yy) != 0:
                for r in range(len(ctrl_yy)):
                    self.ansatz.ryy(yy_theta[targ_yy[r].item()][ctrl_yy[r].item()].item(), ctrl_yy[r].item(), targ_yy[r].item())
            if len(ctrl_zz) != 0:
                for r in range(len(ctrl_zz)):
                    self.ansatz.rzz(zz_theta[targ_zz[r].item()][ctrl_zz[r].item()].item(), ctrl_zz[r].item(), targ_zz[r].item())
            

            thetas = local_state[3*self.n_qubits+3 + 3*self.n_qubits: 3*self.n_qubits+3 + 3*self.n_qubits+3]
            rot_direction_list = one_qubit_pos[0]
            rot_qubit_list = one_qubit_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    # print(r, rot_qubit)
                    if r == 0:
                        self.ansatz.rx(thetas[0][rot_qubit].item(), rot_qubit.item())
                    elif r == 1:
                        self.ansatz.ry(thetas[1][rot_qubit].item(), rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[2][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        
        # print(self.ansatz)
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
        if gate_detail.name in ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']:
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
    
    # circuit.save_statevector()
    # backend = QulacsProvider().get_backend('qulacs_simulator')
    # result = backend.run(circuit).result()
    # state = np.asmatrix(result.get_statevector())
    
    statevector = Statevector(circuit)
    state = np.asmatrix(statevector)
    energy = (state @ op) @ state.getH()

    return float(energy.real)



if __name__ == "__main__":
    pass


















