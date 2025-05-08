import pennylane as qml
import scipy.linalg as la
from pennylane import utils
from pennylane import numpy as np
from qiskit.quantum_info import SparsePauliOp
from pennylane import qchem
import numpy as nnp
# Define molecular geometry and parameters


mol = "H2O"

if mol == 'BEH2':
    symbols = ["H", "Be", "H"]
    geometry = np.array([   [0, 0.0, -1.33],
                            [0, 0, 0],
                            [0, 0, 1.33]
                        ])
    active_electrons = 4
    active_orbitals = 3
    mult = 1
    geom_string = "_".join([f"{symbol}_{',_'.join([f'{coord:.3f}' for coord in coords])}" for symbol, coords in zip(symbols, geometry)])


elif mol == 'H2O':
    symbols = ["H", "O", "H"]
    geometry = np.array([[-0.021, -0.002, .0], [0.835, 0.452, 0], [1.477, -0.273, 0]])
    active_electrons = 4
    active_orbitals = 4
    mult = 1

    # Define the molecule
    symbols = ["H", "O", "H"]
    geometry = np.array([[-0.021, -0.002, 0.0], [0.835, 0.452, 0.0], [1.477, -0.273, 0.0]])

    # Create the formatted string
    geom_string = ";_".join([f"{symbol}_{'_'.join([f'{coord:.3f}' for coord in coords])}" for symbol, coords in zip(symbols, geometry)])

elif mol == 'CH2':
    symbols = ["C", "H", "H"]  # Atomic symbols

    # Bond distance and angle
    bond_distance = 1.08  # C-H bond distance in Angstroms
    bond_angle = np.deg2rad(102)  # H-C-H bond angle in radians

    # Coordinates of the carbon atom
    carbon_coords = np.array([0.0, 0.0, 0.0])

    # Coordinates of the first hydrogen atom (along the x-axis)
    h1_coords = np.array([bond_distance, 0.0, 0.0])

    # Coordinates of the second hydrogen atom (at 102° relative to the first)
    h2_coords = np.array([
        bond_distance * np.cos(bond_angle),
        bond_distance * np.sin(bond_angle),
        0.0
    ])
    mult = 1
    active_electrons = 6  # Number of active electrons
    active_orbitals = 5   # Number of active orbitals

    # Combine into the geometry array
    geometry = np.array([carbon_coords, h1_coords, h2_coords])
    # Create the formatted string
    geom_string = ";_".join([f"{symbol}_{'_'.join([f'{coord:.3f}' for coord in coords])}" for symbol, coords in zip(symbols, geometry)])
    # print(geom_string)
    # exit()

# Create the molecule object
molecule = qml.qchem.Molecule(
    symbols=symbols,
    coordinates=geometry,
    charge=0,
    mult=mult,
    basis_name="sto-3g",
)


# Generate the molecular Hamiltonian
if mol == 'H2O':
    if active_orbitals > 4:
        # Generate the Hamiltonian using a larger basis set
        hamiltonian, qubits = qchem.molecular_hamiltonian(
            symbols,
            geometry,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            # method='pyscf',  # Use 'psi4' if available
            basis='6-31g'    # Larger basis set to support 10 qubits
        )
    else:
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule, active_electrons=active_electrons,
        active_orbitals=active_orbitals)
else:
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule, active_electrons=active_electrons,
        active_orbitals=active_orbitals)


ham = {qml.pauli.pauli_word_to_string(k, wire_map=hamiltonian.wires): v for k, v in zip(hamiltonian.ops, hamiltonian.coeffs)}
paulis_string = list(ham.keys())
mod_weights = list(ham.values())

print("Paulis string:", paulis_string)

complete_dict = dict()
hamiltonian_mat = hamiltonian.matrix()

complete_dict['hamiltonian'] = hamiltonian_mat
complete_dict['eigvals'] = la.eig(hamiltonian_mat)[0].real
complete_dict['weights'] = mod_weights
complete_dict['paulis'] = paulis_string
complete_dict['energy_shift'] = 0

reverse_pauli_str = [p[::-1] for p in paulis_string]

qiskit_mat = SparsePauliOp(paulis_string,
      coeffs= mod_weights)

qiskit_mat_rev = SparsePauliOp(reverse_pauli_str,
      coeffs= mod_weights)


print(len(paulis_string), len(mod_weights))

print(np.linalg.norm(hamiltonian_mat-qiskit_mat.to_matrix()))
print(min(la.eig(hamiltonian_mat)[0].real), min(la.eig(qiskit_mat.to_matrix())[0].real))


if min(la.eig(hamiltonian_mat)[0].real) == min(la.eig(qiskit_mat.to_matrix())[0].real):
    print('HOORAY! Ekdom thik implementation hoyeche!!!')
    print(min(la.eig(hamiltonian_mat)[0].real))

print(geom_string)
print('QUBITS:', qubits)
# exit()
import pickle
with open(f'dmrg-to-qc/mol_data/{mol}_{qubits}q_geom_{geom_string}_jordan_wigner.p', 'wb') as handle:
    pickle.dump(complete_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

nnp.savez(f'dmrg-to-qc/mol_data/{mol}_{qubits}q_geom_{geom_string}_jordan_wigner', **complete_dict)
