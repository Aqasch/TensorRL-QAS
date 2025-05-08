from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import ParametricPauliRotation
import numpy as np


class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.ansatz = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        
        for _, local_state in enumerate(state):
            
            one_qubit_pos = (local_state[3*self.n_qubits: 3*self.n_qubits+3] == 1).nonzero( as_tuple = True )
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
                    RXXgate = self.RXX(ctrl_xx[r].item(), targ_xx[r].item(), xx_theta[targ_xx[r].item()][ctrl_xx[r].item()].item())
                    self.ansatz.add_parametric_gate(RXXgate)
            if len(ctrl_yy) != 0:
                for r in range(len(ctrl_yy)):
                    RYYgate = self.RYY(ctrl_yy[r].item(), targ_yy[r].item(), yy_theta[targ_yy[r].item()][ctrl_yy[r].item()].item())
                    self.ansatz.add_parametric_gate(RYYgate)
            if len(ctrl_zz) != 0:
                for r in range(len(ctrl_zz)):
                    RZZgate = self.RZZ(ctrl_zz[r].item(), targ_zz[r].item(), zz_theta[targ_zz[r].item()][ctrl_zz[r].item()].item())
                    self.ansatz.add_parametric_gate(RZZgate)

            thetas = local_state[3*self.n_qubits+3 + 3*self.n_qubits: 3*self.n_qubits+3 + 3*self.n_qubits+3]
            rot_direction_list = one_qubit_pos[0]
            rot_qubit_list = one_qubit_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    # print(r, rot_qubit)
                    if r == 0:
                        self.ansatz.add_parametric_RX_gate(rot_qubit, thetas[0][rot_qubit])
                    elif r == 1:
                        self.ansatz.add_parametric_RY_gate(rot_qubit, thetas[1][rot_qubit])
                    elif r == 2:
                        self.ansatz.add_parametric_RZ_gate(rot_qubit, thetas[2][rot_qubit])

                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        
        # print(self.ansatz)
        return self.ansatz
    
    def RXX(self, target1, target2, theta):
        """
        CUSTOM RXX
        """
        list_pauli = [1,1]
        gate = ParametricPauliRotation( [target1, target2], list_pauli, theta )
        return gate
    
    def RYY(self, target1, target2, theta):
        """
        CUSTOM RYY
        """
        list_pauli = [2,2]
        gate = ParametricPauliRotation( [target1, target2], list_pauli, theta )
        return gate

    def RZZ(self, target1, target2, theta):
        """
        CUSTOM RZZ
        """
        list_pauli = [3,3]
        gate = ParametricPauliRotation( [target1, target2], list_pauli, theta )
        return gate

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

    parameter_count_qulacs = circuit.get_parameter_count()
    # angles = np.array(angles)
    # angles = jax.device_get(angles)

    
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


















