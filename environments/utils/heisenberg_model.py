import numpy as np
from scipy.sparse import kron, identity, csr_matrix
import scipy.sparse.linalg as la
import numpy as np
from functools import reduce

def construct_hamiltonian(n):
    # Define Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Initialize the Hamiltonian matrix
    hamiltonian = np.zeros((2**n, 2**n), dtype=complex)
    
    # Store Pauli strings and weights
    pauli_strings = []
    weights = []
    
    # Implement sum_{i=1}^{n-1} X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}
    for i in range(n-1):
        # X_i X_{i+1} term
        ops_x = [I] * n
        ops_x[i] = X
        ops_x[i+1] = X
        hamiltonian += reduce(np.kron, ops_x)
        
        # Create Pauli string for X_i X_{i+1}
        pauli_string = ['I'] * n
        pauli_string[i] = 'X'
        pauli_string[i+1] = 'X'
        pauli_strings.append(''.join(pauli_string))
        weights.append(1.0)
        
        # Y_i Y_{i+1} term
        ops_y = [I] * n
        ops_y[i] = Y
        ops_y[i+1] = Y
        hamiltonian += reduce(np.kron, ops_y)
        
        # Create Pauli string for Y_i Y_{i+1}
        pauli_string = ['I'] * n
        pauli_string[i] = 'Y'
        pauli_string[i+1] = 'Y'
        pauli_strings.append(''.join(pauli_string))
        weights.append(1.0)
        
        # Z_i Z_{i+1} term
        ops_z = [I] * n
        ops_z[i] = Z
        ops_z[i+1] = Z
        hamiltonian += reduce(np.kron, ops_z)
        
        # Create Pauli string for Z_i Z_{i+1}
        pauli_string = ['I'] * n
        pauli_string[i] = 'Z'
        pauli_string[i+1] = 'Z'
        pauli_strings.append(''.join(pauli_string))
        weights.append(1.0)
    
    # Implement sum_{i=1}^{n} Z_i
    for i in range(n):
        ops_z = [I] * n
        ops_z[i] = Z
        hamiltonian += reduce(np.kron, ops_z)
        
        # Create Pauli string for Z_i
        pauli_string = ['I'] * n
        pauli_string[i] = 'Z'
        pauli_strings.append(''.join(pauli_string))
        weights.append(1.0)
    
    return hamiltonian, pauli_strings, weights

# Construct the 5-qubit Hamiltonian
n = 5
hamiltonian_matrix, pauli_strings, weights = construct_hamiltonian(n)

# Print results
print(f"Hamiltonian matrix shape: {hamiltonian_matrix.shape}")
print("\nPauli strings and their weights:")
for i, (string, weight) in enumerate(zip(pauli_strings, weights)):
    print(f"{i+1}. {string}: {weight}")

# Create a dictionary mapping Pauli strings to weights
pauli_weights = dict(zip(pauli_strings, weights))
print("\nPauli weights dictionary:")
print(pauli_weights)

complete_dict = dict()
hamiltonian_mat = hamiltonian_matrix #.todense()
mol = 'heisenberg'
qubits = n

complete_dict['hamiltonian'] = hamiltonian_mat
complete_dict['eigvals'] = la.eigs(hamiltonian_mat)[0].real
complete_dict['weights'] = weights
complete_dict['paulis'] = pauli_strings
complete_dict['energy_shift'] = 0

print(min(complete_dict['eigvals']), 'minimum eig')

# print(H_matrix)
print("Hamiltonian matrix shape:", hamiltonian_matrix.shape)
print("Number of terms in Hamiltonian:", len(weights))
print("First few Pauli strings:", pauli_strings[:5])
print("First few weights:", weights[:5])

np.savez(f'dmrg-to-qc/mol_data/{mol}_{qubits}q', **complete_dict)
a = np.load(f'dmrg-to-qc/mol_data/{mol}_{qubits}q.npz', allow_pickle = True)