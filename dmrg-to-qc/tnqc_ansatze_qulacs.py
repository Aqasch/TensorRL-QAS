
import numpy as np
import qiskit as qk
import quimb.tensor as qtn
import scipy as scp
from qulacs.gate import DenseMatrix, RX, RY, RZ, CNOT
from qulacs import ParametricQuantumCircuit, QuantumCircuit, QuantumState
from scipy.linalg import polar, expm

#####################################################################################
# General utility functions
#####################################################################################

def closest_unitary(A):
        """Calculate the unitary matrix U that is closest with respect to the
            operator norm distance to the general matrix A.

            Return U as a numpy matrix.
        """
        V, _, Wh = scp.linalg.svd(A)
        U = V.dot(Wh)
        return U

def qiskit_circ_from_tn_params(ansatz: dict, 
                               params: dict,
                               transpile_opts: dict = {'basis_gates': ['rx', 'ry', 'rz', 'cx'],
                                                       'optimization_level': 3}):
        """Creates a qiskit quantum circuit from the quantum circuit tensor network parameters.
        """
    
        # Enforce unitarity of parameters with more accuracy 
        # otherwise Qiskit complains, see: https://github.com/Qiskit/qiskit/issues/7120
        qc_params = [closest_unitary(p) for p in  params.values()]

        # Create the Qiskit circuit and decompose in basis gates
        state1 = QuantumState(10)
        qc = qulacs_circuit_ansatz(ansatz, qc_params)
        qc.update_quantum_state(state1)
        print(state1.get_vector())
        print(qc)
        print()
        state2 = QuantumState(10)

        decomposed_circuit = decompose_circuit(qc)
        decomposed_circuit.update_quantum_state(state2)
        print(state2.get_vector())
        print(decomposed_circuit)
        exit()
        return qc


#####################################################################################
# Quimb quantum circuit tensor network ansatze
#####################################################################################

def brickwork_ansatz(num_qubits, num_layers, initial_state: qtn.MatrixProductState = None):
    """
    Creates a brickwork ansatz circuit structure for a given number of qubits and layers, evolving an initial state.
    It implements the following circuit in tensor network structure:
          
    |0> --╭──╮--------|--╭──╮--------| ... |
          |G1|        |  |G6|        |     
    |0> --╰––╯--╭──╮--|--╰––╯--╭──╮--| ... |
                |G4|  |        |G9|  |  
    |0> --╭──╮--╰––╯--|--╭──╮--╰––╯--| ... |
          |G2|        |  |G7|        |
    |0> --╰––╯--╭──╮--|--╰––╯--╭──╮--| ... |
                |G5|  |        |G10| |
    |0> --╭──╮--╰––╯--|--╭──╮--╰––╯--| ... |
          |G3|        |  |G8|        |
    |0> --╰––╯--------|--╰––╯--------| ... |
           Layer 1    |   Layer 2    | ... | Layer num_layers

    where |G| can be a general two-qubit unitary, SU(4) gate. Initialized with identities.
          |G|  

    Parameters:
    ------------
    num_qubits (int): The number of qubits in the circuit.
    num_layers (int): The number of layers in the circuit.
    initial_state (qtn.MatrixProductState, optional): The initial state of the circuit (default to |0>^{\\otimes n}).

    Returns:
    --------
    psi (qtn.MatrixProductState): The resulting circuit state after applying the brickwork ansatz.
    counter (int): The total number of gates applied in the circuit.
    """
    
    if initial_state is None:
        initial_state = qtn.MPS_computational_state('0'*num_qubits)

    psi = initial_state.copy()
    psi.add_tag(['INIT_STATE'])

    for q, t in enumerate(psi.tensors):
        t.add_tag(f'K{q}')

    counter = 0
    for _ in range(num_layers):
        for i in range(0, num_qubits-1, 2):
            psi.gate_(np.eye(4), (i, i+1), tags = [f'G{counter}', 'GATE'])
            counter += 1

        for i in range(1, num_qubits-1, 2):
            psi.gate_(np.eye(4), (i, i+1), tags = [f'G{counter}', 'GATE'])
            counter += 1
    psi.add_tag(['KET'])
    return psi, counter


#####################################################################################
# Qiskit quantum circuit ansatze
#####################################################################################

def qulacs_circuit_ansatz(ansatz, params):
    """Dispatch to other functions to create a Qiskit circuit from a given ansatz structure."""
    if ansatz['structure'] == 'brickwork':
        return qulacs_brickwork_ansatz(ansatz['num_qubits'], ansatz['num_layers'], params)
    else:
        raise ValueError(f"Ansatz structure {ansatz['structure']} not implemented for Qiskit circuits.")

def qulacs_brickwork_ansatz(num_qubits, num_layers, params):
    """Creates a parametrized qiskit circuit with a brickwork structure, see above for a graphical representation."""
    # qc = qk.QuantumCircuit(num_qubits)
    qc = ParametricQuantumCircuit(num_qubits)
    
    if isinstance(params, dict):
        params_list = list(params.values())
    else:
        params_list = params

    counter = 0
    # print('--------------')
    # print(params_list[counter])
    # print('--------------')

    # exit()

    for _ in range(num_layers):
        for i in range(0, num_qubits-1, 2):
            # Index of qubit is inverted due to qiskit ordering (I think!)
            gate = DenseMatrix([i+1, i], params_list[counter])
            # qc.unitary(params_list[counter], [i+1, i])
            qc.add_gate(gate)
            counter += 1

        for i in range(1, num_qubits-1, 2):
            gate = DenseMatrix([i+1, i], params_list[counter])

            counter += 1
    
    return qc




def decompose_unitary(U):
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    n = int(np.log2(U.shape[0]))
    circuit = QuantumCircuit(n)
    
    def decompose_2qubit(U):
        if U.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {U.shape}")
        A, U1 = polar(U)
        theta = -np.angle(np.linalg.det(U1)**(1/4))
        U2 = U1 * np.exp(1j * theta)
        
        M = np.array([[1, 0, 0, 1j],
                      [0, 1j, 1, 0],
                      [0, 1j, -1, 0],
                      [1, 0, 0, -1j]]) / np.sqrt(2)
        
        U3 = M.conj().T @ U2 @ M
        # print(np.real(np.diag(U3)))
        diag_values = np.real(np.diag(U3))
        # Handle the case where we have 4 diagonal values instead of 3
        if len(diag_values) == 4:
            alpha, beta, gamma, delta = diag_values
        else:
            alpha, beta, gamma = diag_values
        
        return alpha, beta, gamma, theta, A
    
    U_current = np.eye(2**n, dtype=complex)
    for iteration in range(1000):
        for i in range(n-1):
            for j in range(i+1, n):
                # Extract the 4x4 submatrix for qubits i and j
                indices = []
                for x in range(2**n):
                    if (x & (1 << i) == 0) and (x & (1 << j) == 0):
                        indices.append(x)
                        indices.append(x | (1 << i))
                        indices.append(x | (1 << j))
                        indices.append(x | (1 << i) | (1 << j))
                        break  # We only need one such x
                
                # Extract the 4x4 submatrix
                submatrix = U[np.ix_(indices, indices)]
                
                # Ensure submatrix is 4x4
                if submatrix.shape != (4, 4):
                    raise ValueError(f"Submatrix shape is {submatrix.shape}, expected (4, 4)")
                ax, ay, az, theta, A = decompose_2qubit(submatrix)
                
                circuit.add_gate(RZ(i, az))
                circuit.add_gate(RY(i, ay))
                circuit.add_gate(RZ(i, ax))
                circuit.add_gate(CNOT(i, j))
                circuit.add_gate(RZ(j, -ax))
                circuit.add_gate(RY(j, -ay))
                circuit.add_gate(RZ(j, -az))
                circuit.add_gate(CNOT(i, j))
                
                # Update U with the new submatrix
                decomp = np.kron(expm(1j * (ax*np.kron(sz,sz) + ay*np.kron(sy,sy) + az*np.kron(sx,sx))), np.eye(2**(n-2)))
                U_current = decomp @ U_current

            error = np.linalg.norm(U - U_current)
            if error < 1e-5:
                break

    
        for i in range(n):
            diag_values = np.diag(U)  # Extract the main diagonal of U
            alpha, beta, gamma = np.real(diag_values[i:i+3])  # Extract three
            circuit.add_gate(RZ(i, alpha))
            circuit.add_gate(RY(i, beta))
            circuit.add_gate(RZ(i, gamma))
    
    return circuit


def decompose_circuit(original_circuit):
    n = original_circuit.get_qubit_count()
    new_circuit = QuantumCircuit(n)

    gate_list = [original_circuit.get_gate(i) for i in range(original_circuit.get_gate_count())]

    for gate in gate_list:
        if gate.get_name() == "DenseMatrix":
            matrix = gate.get_matrix()
            decomposed = decompose_unitary(matrix)
            gate_list_decomposed = [decomposed.get_gate(i) for i in range(decomposed.get_gate_count())]
            for decomposed_gate in gate_list_decomposed:
                new_circuit.add_gate(decomposed_gate)
        else:
            new_circuit.add_gate(gate)
    
    return new_circuit


