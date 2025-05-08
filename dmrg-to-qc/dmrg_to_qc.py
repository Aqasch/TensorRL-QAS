import logging
import sys

import jax
import mps2qc
import numpy as np
import tnqc_ansatze as tnqc
from qiskit import qpy, qasm2
from stiefel_opt import StiefelAdam
from qiskit.quantum_info import Statevector, Operator
from qiskit.converters import circuit_to_dag, dag_to_circuit

def get_int_input(prompt, default):
    """Get integer input with default fallback and error handling."""
    try:
        user_input = input(prompt).strip()
        if not user_input:  # Empty input → use default
            return default
        return int(user_input)
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default

def get_mol_config_and_hamiltonian(mol):
    if mol == 'LIH_4q':
        mol_config = "LIH_4q_geom_Li_.0_.0_.0;_H_.0_.0_3.4_parity"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'BEH2_6q':
        mol_config = "BEH2_6q_geom_H_0.000_0.000_-1.330;_Be_0.000_0.000_0.000;_H_0.000_0.000_1.330_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'BEH2_12q':
        mol_config = "BEH2_12q_geom_H_0.000,_0.000,_-1.330_Be_0.000,_0.000,_0.000_H_0.000,_0.000,_1.330_jordan_wigner"
        hamiltonian_data = 'mol_data/BEH2_12q_geom_H_0.000,_0.000,_-1.330_Be_0.000,_0.000,_0.000_H_0.000,_0.000,_1.330_jordan_wigner.npz'
    elif mol == 'H2O_8q':
        mol_config = "H2O_8q_geom_H_-0.021_-0.002_0.000;_O_0.835_0.452_0.000;_H_1.477_-0.273_0.000_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'H2O_10q':
        mol_config = "H2O_10q_geom_H_-0.021_-0.002_0.000;_O_0.835_0.452_0.000;_H_1.477_-0.273_0.000_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'H2O_12q':
        mol_config = "H2O_12q_geom_H_-0.021_-0.002_0.000;_O_0.835_0.452_0.000;_H_1.477_-0.273_0.000_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'CH2_10q':
        mol_config = "CH2_10q_geom_C_0.000_0.000_0.000;_H_1.080_0.000_0.000;_H_-0.225_1.056_0.000_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'CH2_10q_mod':
        mol_config = "CH2_10q_geom_C_0.000_0.000_0.000;_H_1.080_0.000_0.000;_H_-0.225_1.056_0.000mod_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'CH2_8q':
        mol_config = "CH2_8q_geom_C_0.000_0.000_0.000;_H_1.080_0.000_0.000;_H_-0.225_1.056_0.000_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'LIH_full':
        mol_config = "LIH_12q_geom_Li_0.000_0.000_0.000;_H_0.000_0.000_3.400_jordan_wigner"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'heisenberg_12q':
        mol_config = "heisenberg_12q"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'heisenberg_9q':
        mol_config = "heisenberg_9q"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'heisenberg_5q':
        mol_config = "heisenberg_5q"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    elif mol == 'TFIM_6q':
        mol_config = "tfim_j1_h0.05_6q"
        hamiltonian_data = f'mol_data/{mol_config}.npz'
    else:
        mol_config = None
        hamiltonian_data = None
    return mol_config, hamiltonian_data

def molecule_choice(mol_list):
    print("Available molecules:")
    for idx, name in enumerate(mol_list):
        print(f"{idx}: {name}")
    try:
        user_input = int(input(f"Select a molecule (0-{len(mol_list)-1}): "))
        if not (0 <= user_input < len(mol_list)):
            print("Invalid input. Please enter a number within the valid range.")
            return
        mol = mol_list[user_input]
        mol_config, hamiltonian_data = get_mol_config_and_hamiltonian(mol)
        if mol_config is None:
            print("Configuration not found for the selected molecule.")
        else:
            print(f"Selected molecule: {mol}")
            print(f"mol_config: {mol_config}")
            print(f"hamiltonian_data: {hamiltonian_data}")
            return mol_config, hamiltonian_data
    except ValueError:
        print("Invalid input. Please enter an integer.")

def trimmed_circuit(qc, max_depth):
    """
    Trim a quantum circuit to a maximum depth using its DAG representation.

    Args:
        qc (QuantumCircuit): The input quantum circuit.
        max_depth (int): The maximum allowed depth.

    Returns:
        QuantumCircuit: The trimmed quantum circuit.
    """
    # Convert the circuit to a DAG representation
    dag = circuit_to_dag(qc)
    
    # Create a new empty DAG to store the trimmed circuit
    trimmed_dag = dag.copy_empty_like()
    
    # Iterate through layers of the DAG
    layers = list(dag.layers())
    for i, layer in enumerate(layers):
        if i >= max_depth:
            break  # Stop adding layers once we reach `max_depth`
        for node in layer['graph'].nodes():
            # Check if the node is an operation node (DAGOpNode)
            if isinstance(node, dag.op_nodes().__iter__().__next__().__class__):  # Dynamically check for DAGOpNode type
                trimmed_dag.apply_operation_back(node.op, node.qargs, node.cargs)
    
    # Convert the trimmed DAG back to a QuantumCircuit
    trimmed_circuit = dag_to_circuit(trimmed_dag)
    
    return trimmed_circuit

jax.config.update("jax_enable_x64", True)

logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s [%(module)s] %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S',
                    level=logging.INFO)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

DIV_STR = "-"*80

def main(hamiltonian_path: str,
         dmrg_opts: dict, 
         ansatz: dict, 
         optimizer_opts: dict,
         su4: int):
    """
    Main function to perform DMRG and convert the resulting MPS to a quantum circuit.
    Parameters:
        hamiltonian_path (str): Path to the file containing the Hamiltonian data in 'npz' format.
        dmrg_opts (dict): Options for the DMRG algorithm (using quimb's DMRG).
        ansatz (dict): Ansatz configuration for the quantum circuit.
        optimizer_opts (dict): Options for the optimizer used in the MPS to QC conversion.
    Returns:
        dict: A dictionary containing the following keys:
            - 'psi_dmrg': The ground state MPS obtained from DMRG.
            - 'dmrg_metadata': Metadata from the DMRG run.
            - 'qc_mps': The optimized quantum circuit tensor network.
            - 'opt_params': Optimized parameters for the quantum circuit.
            - 'loss_history': History of the loss function during optimization.
            - 'qc': The final Qiskit quantum circuit.
    """

    # Extract the Hamiltonian from the data
    data = np.load(hamiltonian_path)
    # print(data.keys())
    # exit()
    # The [::-1] is probably because of qiskit's ordering when saving Pauli strings?
    # It is needed to get the correct Hamiltonian comparing with the one in the data
    # pauli_dict = {k[::-1]: np.real_if_close(v) for k, v in zip(data['paulis'], data['weights'])}
    pauli_dict = {k: np.real_if_close(v) for k, v in zip(data['paulis'], data['weights'])}

    # Construct the MPO of the Hamiltonian
    ham_mpo, _ = mps2qc.mpo_from_paulis(pauli_dict)

    # Extract number of qubits from the provided Hamiltonian
    num_qubits = ham_mpo.L

    # Check that the MPO hamiltonian is correct
    if num_qubits < 10:
        logging.info(f" ║ H_dense - H_mpo ║  = {np.linalg.norm(ham_mpo.to_dense() - data['hamiltonian'])}")
        
    
    logging.info(DIV_STR)
    logging.info("Running DMRG")
    psi_dmrg, dmrg_metadata = mps2qc.gs_dmrg(ham_mpo, **dmrg_opts)
    logging.info(DIV_STR)
    
    logging.info(DIV_STR)
    logging.info("Fitting ansatz quantum circuit TN to DMRG-MPS")
    qc_mps, loss_history, opt_params = mps2qc.mps_to_qc(psi_dmrg, 
                                                        ansatz=ansatz,
                                                        optimizer_opts=optimizer_opts)   
    logging.info(DIV_STR)

    logging.info(DIV_STR)
    logging.info(f"Energy [dmrg] 〈H〉= {mps2qc.compute_energy(psi_dmrg, ham_mpo)}")
    logging.info(f"Energy [qc-tn]〈H〉= {mps2qc.compute_energy(qc_mps, ham_mpo)}")
    logging.info(f"Energy [qc-tn] - Energy [dmrg] 〈H〉= {abs(mps2qc.compute_energy(qc_mps, ham_mpo) - mps2qc.compute_energy(psi_dmrg, ham_mpo))}")

    logging.info(DIV_STR)
    
    logging.info(DIV_STR)
    logging.info("Building qiskit quantum circuit")
    ansatz['num_qubits'] = num_qubits

    # print(ansatz)
    if su4:
        basis_gates = ['rxx', 'ryy', 'rzz', 'rx', 'ry', 'rz']
    else:
        basis_gates = ['cx', 'rx', 'ry', 'rz']

    qc = tnqc.qiskit_circ_from_tn_params(ansatz, 
                                         opt_params,
                                         transpile_opts={'optimization_level': 3,
                                                         'basis_gates': basis_gates})
    logging.info(DIV_STR)
    # exit()
    
    
    results = {'psi_dmrg': psi_dmrg,
               'dmrg_metadata': dmrg_metadata,
               'qc_mps': qc_mps,
               'opt_params': opt_params,
               'loss_history': loss_history,
               'qc': qc}
    
    return results

if __name__ == '__main__':

    # List of molecules
    mol_list = [
        'heisenberg_5q','LIH_4q', 'BEH2_6q', 'H2O_8q', 'H2O_12q', 'CH2_10q',
        'CH2_10q_mod', 'H2O_10q', 'LIH_full', 'BEH2_12q', 'heisenberg_12q',
        'heisenberg_9q', 'TFIM_6q', 'CH2_8q'
    ]

    mol_config, hamiltonian_data = molecule_choice(mol_list)

    """
    TO CHOOSE MORE EXPRESSIVE GATESET CONTAININ XX, YY, ZZ
    """
    SU4 = 0

    """
    DMRG OPT AND MPS --> PQC PARAMETERS
    """

    bond_dims = get_int_input("Choose bond dimension for DMRG (default 2): ", 2)
    num_layers = get_int_input("Choose layers in PQC (default 1): ", 1)
    print('-x-x-x-x-x-x-x-x-')
    print(f'DMRG is genarating MPS with bond dimension {bond_dims}')
    print(f'MPS is transformed into a PQC with layers {num_layers}')
    print('-x-x-x-x-x-x-x-x-')
    print()

    dmrg_opts = {'bond_dims': [bond_dims],
                 'num_sweeps': 2}

    qc_tn_ansatz = {'structure': 'brickwork',
                    'num_layers': num_layers}

    optimizer = StiefelAdam(learning_rate=3e-3,
                            beta1=0.9,
                            beta2=0.999,
                            eps=1e-8)    

    optimizer_opts = {'method': optimizer,
                      'maxiter': 2000,
                      'tol': 1e-8}
    
    res = main(hamiltonian_data, dmrg_opts, qc_tn_ansatz, optimizer_opts, SU4)

    TN_init_circuit = res['qc']
    """
    MPS ->> PQC (drawing):
    """
    print(TN_init_circuit)
    print('-------------')
    print()
    """
    Resources in PQC:
    """
    print('Gates:', TN_init_circuit.count_ops(), 'Depth:', TN_init_circuit.depth())



    data = np.load(hamiltonian_data, allow_pickle = True)
    state = Statevector(TN_init_circuit)
    ham = data['hamiltonian']
    ham = Operator(ham).reverse_qargs().to_matrix()
    state = np.asmatrix(state)
    energy_qiskit = (state @ ham) @ state.getH()
    
    TNbond = dmrg_opts['bond_dims'][0]
    if SU4:
        qasm2.dump(TN_init_circuit, f"init_state_circ/init_{mol_config}_TNbond{TNbond}_su4.qasm")
        with open(f"init_state_circ/init_{mol_config}_TNbond{TNbond}_su4.qpy", "wb") as file:
            qpy.dump(TN_init_circuit, file)
        loaded_circuit = qasm2.load(f'init_state_circ/init_{mol_config}_TNbond{TNbond}_su4.qasm', custom_instructions = qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    else:
        qasm2.dump(TN_init_circuit, f"init_state_circ/init_{mol_config}_TNbond{TNbond}.qasm")
        with open(f"init_state_circ/init_{mol_config}_TNbond{TNbond}.qpy", "wb") as file:
            qpy.dump(TN_init_circuit, file)
        loaded_circuit = qasm2.load(f'init_state_circ/init_{mol_config}_TNbond{TNbond}.qasm', custom_instructions = qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    
    state = Statevector(loaded_circuit)
    state = np.asmatrix(state)
    energy = (state @ ham) @ state.getH()
    print()
    print()
    print('-X-X-X-X-X- SOME SANITY CHECK -X-X-X-X-X-')
    print('THE OBTAINED ENERGY FROM (QISKIT) CIRCUIT: ', energy_qiskit.real)
    print('THE OBTAINED ENERGY FROM LOADED CIRCUIT: ', energy.real)
    # print(energy_qiskit.real - energy.real)
    if np.abs(energy_qiskit.real - energy.real) <= 1e-6:
        print('They are same! Everything is working as it is supposed to be!')
    else:
        print('Someting is not right!')

    