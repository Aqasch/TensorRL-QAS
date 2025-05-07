import logging

import jax
import numpy as np
import qiskit as qk
import quimb.tensor as qtn
import scipy as scp
import tnqc_ansatze as tnqc
from jax import numpy as jnp
from jax.tree_util import tree_map
from more_itertools import first

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

def rand_uni(n: int):
    """Returns a random unitary matrix of dimension n x n."""
    return scp.stats.unitary_group.rvs(n)

#####################################################################################
# Tensor network utility functions
#####################################################################################
def mpo_from_paulis(
    hamiltonian: dict[str, complex],
) -> tuple[qtn.MatrixProductOperator, float]:
    """Construct an MPO from an operator expressed as a sum of Pauli strings.

    Args:
        hamiltonian: the Hamiltonian from which to construct the MPO

    Returns:
        a tensor network with ket, bra indices
        the norm of the vector of coefficients in the operator
    """
    id = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1,0],[0,-1]])
    Pauli = {'I': id, 'X': x, 'Y': y, 'Z': z}

    num_qubits = len(first(hamiltonian))
    num_ops = len(hamiltonian)
    norm = 0.0

    cores = []

    q = 0
    data = np.zeros((2, 2, num_ops), dtype=complex)
    for ps_ind, (ps, coeff) in enumerate(hamiltonian.items()):
        norm += np.abs(coeff) ** 2
        data[:, :, ps_ind] = Pauli[ps[q]]
        if ps_ind % num_qubits == q:
            data[:, :, ps_ind] *= coeff
    cores.append(data)

    for q in range(1, num_qubits - 1):
        data = np.zeros((2, 2, num_ops, num_ops), dtype=complex)
        for ps_ind, (ps, coeff) in enumerate(hamiltonian.items()):
            data[:, :, ps_ind, ps_ind] = Pauli[ps[q]]
            if ps_ind % num_qubits == q:
                data[:, :, ps_ind, ps_ind] *= coeff
        cores.append(data)

    q = num_qubits - 1
    data = np.zeros((2, 2, num_ops), dtype=complex)
    for ps_ind, (ps, coeff) in enumerate(hamiltonian.items()):
        data[:, :, ps_ind] = Pauli[ps[q]]
        if ps_ind % num_qubits == q:
            data[:, :, ps_ind] *= coeff
    cores.append(data)

    norm *= 1 << num_qubits

    return qtn.MatrixProductOperator(arrays=cores, shape="udlr"), norm

def compute_energy(psi, hamiltonian):
    """Computes the energy of a given state with respect to a Hamiltonian.
    Both of them are represented as tensor networks."""
    b, h, k = qtn.tensor_network_align(psi.H, hamiltonian, psi)
    return (b | h | k).contract(optimize = 'auto-hq', backend = 'jax')

#####################################################################################
# Ground state MPS solvers
#####################################################################################
def gs_exact_diag(hamiltonian: np.ndarray) -> float:
    """
    Compute the ground state energy of a Hamiltonian using exact diagonalization.
    Of course, works only for small systems.
    
    Parameters:
    -----------
    hamiltonian (np.ndarray): The Hamiltonian matrix.
    
    Returns:
    --------
    float: The ground state energy.
    """
    eigvals, _ = np.linalg.eigh(hamiltonian)
    return np.min(eigvals)

def gs_dmrg(hamiltonian_mpo: qtn.MatrixProductOperator,
            bond_dims: list[int] = [2],
            num_sweeps: int = 5,
            verbose: bool = False) -> qtn.MatrixProductState:
    """
    Compute the ground state of a given Hamiltonian using the Density Matrix Renormalization Group (DMRG) algorithm
    as implemented in `quimb` using `quimb.tensor.dmrg` function.
    
    Parameters:
    -----------
    hamiltonian_mpo (qtn.MatrixProductOperator): 
        The Hamiltonian represented as a Matrix Product Operator (MPO).
    bond_dims : list[int], optional
        List of bond dimensions to use during the DMRG sweeps. Default is [2].
    verbose : bool, optional
        If True, visualizes the tensor network energy during the process. Default is False.
    
    Returns:
    --------
    qtn.MatrixProductState
        The ground state of the Hamiltonian represented as a Matrix Product State (MPS).
    """
    
    # Initialize DMRG instance
    dmrg = qtn.DMRG(hamiltonian_mpo, bond_dims=bond_dims, cutoffs=0.) # CUTOFF IS DEFAULT = 0

    if verbose:
        dmrg.TN_energy.draw(color = ['_KET', '_HAM', '_BRA'], show_tags=False)

    # Run DMRG
    dmrg.solve(max_sweeps=num_sweeps, verbosity=1)
    dmrg_energy = dmrg.energy
    dmrg_mps = dmrg.state

    if hamiltonian_mpo.L < 16:
        logger.info("Computing ground state energy from exact diagonalization of the hamiltonian")
        gs_ed = gs_exact_diag(hamiltonian_mpo.to_dense())

    logger.info(f"MPS bond dimensions           = {dmrg_mps.bond_sizes()}")
    logger.info(f"Energy [dmrg]                 = {dmrg_energy}")
    
    if hamiltonian_mpo.L < 16:
        logger.info(f"Energy [ground state]         = {np.min(gs_ed)}")
        err = np.abs(dmrg_energy - np.min(gs_ed))
        logger.info(f"|Energy [dmrg] - Energy [gs]| = {err}")
        chem_accuracy = 1.6e-3
        if err < chem_accuracy:
            logger.info(f"DMRG energy is within chemical accuracy: {err} < {chem_accuracy}!")
        else:
            logger.info(f"DMRG energy is NOT within chemical accuracy: {err} > {chem_accuracy}!")
    
    return dmrg_mps, dmrg


def gs_autodiff(hamiltonian_mpo: qtn.MatrixProductOperator,
                bond_dim: int = 2,
                opt_steps: int = 5000,
                optimizer: str = "l-bfgs-b") -> qtn.MatrixProductState:
    """
    Solve for the ground state of a Hamiltonian represented as a Matrix Product Operator (MPO)
    by minimizing the global energy via automatic differentiation of a Matrix Product State (MPS).
    
    If the number of qubits is less than 10, the function will also compute the ground-truth
    ground state energy using exact diagonalization for comparison.
    
    Parameters:
    -----------
    hamiltonian_mpo (qtn.MatrixProductOperator): The Hamiltonian represented as a Matrix Product Operator.
    bond_dim (int): The bond dimension of the Matrix Product State (default is 2).
    opt_steps (int): The number of optimization steps to perform (default is 5000).
    optimizer (str): The optimization algorithm to use (default is "l-bfgs-b").
    verbose (bool) : If True, plot the optimization progress and print additional information (default is False).
    
    Returns:
    --------
    qtn.MatrixProductState: The optimized Matrix Product State representing the ground state.
    
    """

    def norm_fn(psi):
        nfact = (psi.H @ psi) ** 0.5
        return psi.multiply(1 / nfact, spread_over='all')

    def loss_fn(psi, ham):
        b, h, k = qtn.tensor_network_align(psi.H, ham, psi)
        energy_tn = b | h | k
        return (energy_tn ^ ...).real
    
    num_qubits = hamiltonian_mpo.L

    # Initialize the MPS with a random state
    psi = qtn.MPS_rand_state(num_qubits, 
                             bond_dim=bond_dim,
                             dtype = 'complex128')

    # Define the optimzier
    tnopt = qtn.TNOptimizer(psi,
                            loss_fn=loss_fn,
                            norm_fn=norm_fn,
                            loss_constants={"ham": hamiltonian_mpo},
                            optimizer=optimizer, # 'adam', 'l-bfgs-b',
                            autodiff_backend="jax")

    # Run the optimization
    psi_opt = tnopt.optimize(opt_steps)

    logger.info(f"GS energy (tn-autodiff) = {tnopt.loss_best}")
    logger.info(f"MPS bond dimensions  {psi_opt.bond_dims()}")
    if num_qubits < 14:
        logger.info("Computing energy from exact diagonalization of the hamiltonian...")
        gs_ed = gs_exact_diag(hamiltonian_mpo.to_dense())
        logger.info(f"GS energy (exact diag) = {gs_ed}")
        logger.info(f"Error: {np.abs(tnopt.loss_best - np.min(gs_ed))}")
    
    return psi_opt, tnopt


#####################################################################################
# Quantum circuit tensor network optimization
#####################################################################################
def build_overlap_tn(mps_target, mps_ansatz):
    """Build the overlap tensor network <mps_target|mps_ansatz> 
    between the target MPS and the variational quantum circuit tensor network.
    
    Parameters:
    -----------
    mps_target (qtn.MatrixProductState): The target matrix product state.
    mps_ansatz (qtn.TensorNetwork): The ansatz tensor network quantum circuit to be optimized.
    
    Returns:
    --------
    qtn.TensorNetwork: The tensor network representing the overlap.

    """
    mps = mps_target.copy()
    mps.add_tag(['TARGET', 'BRA'])
    [t.add_tag(f'B{q}') for q, t in enumerate(mps.tensors)]
    bra, ket = qtn.tensor_network_align(mps.H, mps_ansatz)
    return (bra | ket)

def mps_to_qc(mps: qtn.MatrixProductState, 
              ansatz: dict = {'structure': 'brickwork',
                              'num_layers': 2},
              optimizer_opts: dict = None,              
              verbose: bool = False) -> qk.QuantumCircuit:
    """
    Finds a quantum circuit with given structure that is close to a target Matrix Product State (MPS).
    This is done via a tensor network representation of the circuit, using automatic differentiation and stiefel optimization.
    
    Parameters:
    -----------
    mps (qtn.MatrixProductState): 
        The Matrix Product State to be mapped to a quantum circuit. 
    
    Returns:
    --------
    qtn.TensorNetwork: 
        The resulting tensor network representing the quantum circuit.
    """

    def mats_to_tensors(mats):
        """Reshape (4, 4) matrices to (2,2,2,2) tensors."""
        return tree_map(lambda m: jnp.reshape(m, (2,2,2,2)), mats)
    
    num_qubits = mps.L

    # Create the tensor network representing the quantum circuit ansatz 
    structure = ansatz.get('structure', None)
    if structure == 'brickwork':
        num_layers = ansatz.get('num_layers', None)    
        qc_mps, num_gates = tnqc.brickwork_ansatz(num_qubits, num_layers)
    else:
        raise ValueError("Unknown ansatz type. Only 'brickwork' is supported.")
    
    # Create tensor network for the overlap <mps|qc_mps>
    overlap_tn = build_overlap_tn(mps, qc_mps)

    # Select only tensors corresponding to gates to be optimized
    # this is a dictionary of the form {tensor_number: array}
    params = overlap_tn.select('GATE').get_params()
    
    ############################################################
    # Optimization starts
    ############################################################
    # Loss function to be minimized
    @jax.jit
    def loss_fn(params: jnp.array):
        """
        Loss function to minimize the overlap between the ansatz and target mps.
        L(params) = 1 - |<qc(params)|mps>|
        """
        params = mats_to_tensors(params)
        overlap_tn.set_params(params)
        overlap = overlap_tn.contract(optimize = 'auto-hq',
                                      backend = 'jax')
        return 1 - jnp.abs(overlap)

    # Set initial 2-qubit gates in circuit as random 2-qubit unitaries
    init_params = {k: jnp.array(rand_uni(4)) for k in params.keys()}

    # Extract optimizer options
    optimizer = optimizer_opts.get('method', None)
    max_iter  = optimizer_opts.get('max_iter', 2_000)
    tol = optimizer_opts.get('tol', 1e-6)
    param_tol = optimizer_opts.get('param_tol', 1e-6)
    
    # Initialize optimizer 
    optimizer.init(init_params)

    # Minimize loss
    optimizer.minimize(loss_fn, 
                       init_params, 
                       max_iter = max_iter, 
                       tol = tol, 
                       param_tol = param_tol)
    
    # Get the final results
    opt_params = optimizer.opt_params
    loss_history = optimizer.loss_history

    # Extract optimal parameters and update the quantum circuit tensor network
    overlap_tn.set_params(mats_to_tensors(opt_params))
    
    # Fix mismatch in parameter names between qc_mps and overlap_tn
    qc_params = {k: v for k, v in zip(qc_mps.get_params().keys(), 
                                      overlap_tn.select('KET').get_params().values())}
    
    # Set the optimal parameters in the quantum circuit tensor network
    qc_mps.set_params(qc_params)

    # Log results
    # Loss function is phase invariant. One can extract the phase difference 
    # between the optimized and target MPS from their overlap
    phase = overlap_tn.contract(optimize = 'auto-hq')
    logger.info(f"Overlap:  │〈mps|qc_tn〉│             = {1 - np.min(loss_history)}")
    logger.info(f"Distance: ║ phase * |mps〉- |qc_tn〉║ = {mps.multiply(phase).distance(qc_mps)}")

    return qc_mps, loss_history, opt_params






    