
import numpy as np
import qiskit as qk
import quimb.tensor as qtn
import scipy as scp

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
                                                    #    'basis_gates': ['ry', 'cx'],
                                                       'optimization_level': 3}):
        """Creates a qiskit quantum circuit from the quantum circuit tensor network parameters.
        """
    
        # Enforce unitarity of parameters with more accuracy 
        # otherwise Qiskit complains, see: https://github.com/Qiskit/qiskit/issues/7120
        qc_params = [closest_unitary(p) for p in  params.values()]

        # Create the Qiskit circuit and decompose in basis gates

        qc = qiskit_circuit_ansatz(ansatz, qc_params)
        

        return qk.transpile(qc, **transpile_opts)
        # return qc


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

def qiskit_circuit_ansatz(ansatz, params):
    """Dispatch to other functions to create a Qiskit circuit from a given ansatz structure."""
    if ansatz['structure'] == 'brickwork':
        return qiskit_brickwork_ansatz(ansatz['num_qubits'], ansatz['num_layers'], params)
    else:
        raise ValueError(f"Ansatz structure {ansatz['structure']} not implemented for Qiskit circuits.")

def qiskit_brickwork_ansatz(num_qubits, num_layers, params):
    """Creates a parametrized qiskit circuit with a brickwork structure, see above for a graphical representation."""
    qc = qk.QuantumCircuit(num_qubits)
    
    if isinstance(params, dict):
        params_list = list(params.values())
    else:
        params_list = params

    counter = 0
    for _ in range(num_layers):
        for i in range(0, num_qubits-1, 2):
            # Index of qubit is inverted due to qiskit ordering (I think!)
            qc.unitary(params_list[counter], [i+1, i], label = f'G{counter}') 
            counter += 1

        for i in range(1, num_qubits-1, 2):
            qc.unitary(params_list[counter], [i+1, i], label = f'G{counter}')
            counter += 1
    
    return qc