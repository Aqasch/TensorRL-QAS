from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import CNOT
from qulacs.gate import *
import numpy as np

class Parametric_Circuit:
    
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]


            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    
                    if r == 0:
                        self.ansatz.add_parametric_RX_gate(rot_qubit, thetas[0][rot_qubit])
                    elif r == 1:
                        self.ansatz.add_parametric_RY_gate(rot_qubit, thetas[1][rot_qubit])
                    elif r == 2:
                        self.ansatz.add_parametric_RZ_gate(rot_qubit,  thetas[2][rot_qubit])
                    else:
                        print('The angle is not right')
        
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
    

    parameter_count_qulacs = circuit.get_parameter_count()
    
    if not list(which_angles):
        which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    expval = get_exp_val(n_qubits,circuit,observable)
    return expval

def get_exp_val(n_qubits,circuit,op):

    state = QuantumState(n_qubits)
    
    circuit.update_quantum_state(state)
    psi = state.get_vector()
    expval = (np.conj(psi).T @ op @ psi).real
    return expval

if __name__ == "__main__":
    pass


















