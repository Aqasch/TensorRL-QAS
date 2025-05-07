import logging
import operator
from typing import Callable

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

class StiefelOptimizer():
    """Riemannian optimization on the Stiefel Manifold of *Unitary* matrices.
    
    Note:
        Apart for 1/2 factors, for square matrices the euclidean and canonical metric 
        on the Stiefel manifold coincide, hence also the respective gradients. 
        See https://arxiv.org/abs/physics/9806030.

    """

    def __init__(self, 
                 learning_rate: float,
                 opt_state: dict = {}):
        """
        Args:
           learning_rate: The learning rate.
           opt_state: The optimizer state, containing info on the current iteration.
        """

        self.learning_rate = learning_rate
        self.opt_state = opt_state # this is defined in specific optimizers.

    def _riemannian_grad(self, grad, param):
        """Computation of the Riemannian gradient, as in Eq. (14) from https://arxiv.org/pdf/2007.01287.
        
        Note:
            For square unitary matrices, it matches the Eculidean metric gradient.
        """
        return grad - param @ grad.T.conj() @ param
    
    def _vec_riemannian_grad(self, grads, params):
        """Compute the Riemannian gradient of a list of matrices."""
        return tree_map(self._riemannian_grad, grads, params)

    def _cayley_retraction(self, grad, param):
        """Computation of the Cayley retraction, as in Eq. (15) from https://arxiv.org/pdf/2007.01287.
        
        Note: 
            For unitary matrices the formula simplifies significantly.
        """
        a = grad @ param.T.conj() - param @ grad.T.conj()
        b = jnp.linalg.inv(jnp.eye(4) - 0.5 * a)
        c =               (jnp.eye(4) + 0.5 * a)
        return b @ c @ param
    
    def _vec_cayley_retraction(self, grads, params):
        """Compute the Cayley retraction to a list of matrices."""
        return tree_map(self._cayley_retraction, grads, params)
    
    def _vector_transport(self, grad, param):
        """Vector transport as in Eq. (16) from https://arxiv.org/pdf/2007.01287.
        
        Note: 
            For unitary matrices the formula simplifies significantly, matching the riemaniann gradient,
            up to a factor of 0.5.
        """
        return 0.5 * self._riemannian_grad(grad, param)
    
    def _vec_vector_transport(self, grads, params):
        """Compute vector transport to a list of matrices."""
        return tree_map(self._vector_transport, grads, params)
    
    def _metric(self, X, A, B):
        """
        Metric on the Stiefel manifold, see pag. 7 in https://arxiv.org/pdf/2007.01287.

        Note:
            For unitary matrices, the canonical and euclidean metric coincide. In particular, 
            there is no dependence on the manifold position X. Keeping it for consistency with formulas. 
        """
        return jnp.real(jnp.trace(A.T.conj() @ B))
        
    def update(self, params, grads, opt_state):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def minimize(self, 
                 loss_fn: Callable[[jnp.ndarray], jnp.ndarray], 
                 init_params: jnp.ndarray,
                 max_iter: int = 1000, 
                 tol: float = 1e-10,
                 param_tol: float = 1e-6):
        """
        """

        @jax.jit
        def step(params):
            """
            Full update step of the optimization algorithm: (euclidean) gradient computation + parameter update
            """
            self.opt_state['iter'] += 1
            val, grads = val_and_grad_fn(params)
            grads = tree_map(lambda x: x.conj(), grads)
            params, self.opt_state = self.update(params, 
                                                 grads, 
                                                 self.opt_state)
            return val, params, self.opt_state
        
        @jax.jit
        def mean_param_diff(params, old_params): 
            """Utility function to compute the difference between two sets of parameters."""                       
            diff = tree_map(lambda x, y: jnp.linalg.norm(x - y), params, old_params)
            return tree_reduce(operator.add, diff) / len(diff)

        params = init_params.copy()

        # Compile the loss function and its gradient
        val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Arbitrarily large value so that first step becomes best_val
        best_val = 10000 
        
        logger.info(f"Starting optimization with {max_iter=}, {tol=}, {param_tol=}.")
        
        loss_history = []
        progbar = tqdm(range(max_iter))
        for _ in progbar:
            old_params = params.copy() # save old parameters for comparison with new ones
            val, params, self.opt_state = step(params) # full gradient update step
            loss_history.append(val.tolist()) # save loss history
            
            if val < best_val: # store best value so far
                best_val = val
                best_params = params.copy()
            
            # Update progress bar with current metrics
            progbar.set_description_str(f"Step {_ + 1} / {max_iter} — best loss: {best_val} — loss: {val}")

            # Check if loss is below tolerance
            if val < tol:
                progbar.close()
                logger.info(f"Converged to the desired tolerance = {tol}, terminating.")
                break
            
            # Check if parameters are still changing sensibly
            if mean_param_diff(params, old_params) < param_tol:
                progbar.close()
                logger.info(f"|| params_new - params_old ||_2 < {param_tol}, terminating.")
                break
        
        self.best_val = best_val
        self.opt_params = best_params
        self.loss_history = loss_history

class StiefelGD(StiefelOptimizer):
    """
    Implements standard gradient descent (GD) on the Stiefel manifold, as described in https://arxiv.org/pdf/2007.01287.
    """

    def __init__(self, 
                 learning_rate: float = 1e-3,
                 opt_state: dict = {}):

        super().__init__(learning_rate, opt_state)

    def init(self, params):
        """Initialize the optimizer state and hyperparameters."""
        # Cast learning rate as a dictionary (same type of parameters)
        self.lr = {k: self.learning_rate for k in params.keys()}
        
        # Initialize optimizer state to 0 iterations
        self.opt_state['iter'] = 0
    
    def update(self, params, grads, opt_state):
        """Performs a single Gradient Descent (GD) update step.
        
        Args:
            params (array-like): Current parameters of the model.
            grads (array-like): Euclidean gradients of the loss with respect to the parameters.
            opt_state (dict): State of the optimizer containing the momentum.
        
        Returns:
            tuple: Updated parameters and optimizer state.
        
        The optimization step involves the following steps:
            1. Reshape the parameters and gradients from (2,2,2,2) to (4,4).
            2. Compute the Riemannian gradient at current point on manifold.
            4. Update the parameters using the Cayley retraction.
            6. Reshape the parameters back to (2,2,2,2).
        """
        
        # Compute Riemaniann gradient
        riemannian_grads = self._vec_riemannian_grad(grads, params)

        # Move to new point on manifold via Cayley retraction
        update_direction = tree_map(lambda eta, p: - eta * p, self.lr, riemannian_grads)
        params = self._vec_cayley_retraction(update_direction, params)

        return params, opt_state

class StiefelMomentumGD(StiefelOptimizer):
    """Momentum gradient descent on the Stiefel manifold, as described in https://arxiv.org/pdf/2007.01287."""

    def __init__(self, 
                 learning_rate: float = 1e-1, 
                 beta: float = 0.9,
                 opt_state: dict = {}):
        
        super().__init__(learning_rate, opt_state)
        self.beta = beta

    def init(self, params):
        """Initialize the optimizer state and hyperparameters."""
        # Cast learning rate as a dictionary (same type of parameters)
        self.lr = {k: self.learning_rate for k in params.keys()}
        
        # Initialize optimizer to 0 iterations
        self.opt_state['iter'] = 0

        # Initialize momentum to zeros
        self.opt_state['momentum'] = {k: jnp.zeros_like(v, dtype=complex) for k, v in params.items()}

    def update(self, params, grads, opt_state):
        """Performs a single optimization step.
        
        Args:
            params (array-like): Current parameters of the model.
            grads (array-like): Euclidean gradients of the loss with respect to the parameters.
            opt_state (dict): State of the optimizer containing the momentum.
        
        Returns:
            tuple: Updated parameters and optimizer state.
        
        The optimization step involves the following steps:
            1. Reshape the parameters and gradients from (2,2,2,2) to (4,4).
            2. Compute the Riemannian gradient at current point on manifold.
            3. Compute the update direction using momentum.
            4. Update the parameters using the Cayley retraction.
            5. Transport the momentum to the new point on the manifold.
            6. Reshape the parameters back to (2,2,2,2).
        """

        # Compute Riemaniann gradient
        riemannian_grads = self._vec_riemannian_grad(grads, params)

        # Compute update direction given momentum
        momentum = tree_map(lambda m, rg: self.beta * m + (1 - self.beta) * rg, 
                            opt_state['momentum'],
                            riemannian_grads)

        # Move to new point on manifold via Cayley retraction
        update_direction = tree_map(lambda eta, p: - eta * p, self.lr, momentum)
        params = self._vec_cayley_retraction(update_direction, params)

        # Transport moment to new point on manifold
        opt_state['momentum'] = self._vec_vector_transport(momentum, params)

        return params, opt_state


class StiefelAdam(StiefelOptimizer):
    """
    Implements Adam on the Stiefel manifold, as described in https://arxiv.org/pdf/2007.01287.

    Note:
        See other similar code: 
        - https://github.com/INMLe/rqcopt-mpo/blob/main/rqcopt_mpo/adam.py
        - https://github.com/LuchnikovI/QGOpt/tree/master/QGOpt/optimizers
    """

    def __init__(self, 
                 learning_rate: float = 1e-1, 
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 eps: float = 1e-10,
                 opt_state: dict = {}):
        
        super().__init__(learning_rate, opt_state)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init(self, params):
        """Initialize the optimizer state and hyperparameters."""
        # Cast learning rate as a dictionary (same type of parameters)
        self.lr = {k: self.learning_rate for k in params.keys()}
        
        # Initialize optimizer to 0 iterations
        self.opt_state['iter'] = 0

        # Initialize momentum and velocity to zeros
        self.opt_state['momentum'] = {k: jnp.zeros_like(v, dtype=complex) for k, v in params.items()}
        self.opt_state['velocity'] = {k: jnp.zeros_like(v, dtype=complex) for k, v in params.items()}
    
    def update(self, params, grads, opt_state):
        """Perform a single optimization step on the given parameters.
        
        Args:
            params (dict): The parameters to optimize.
            grads (dict): The gradients of the parameters.
            opt_state (dict): Dictionary containing the current state of the optimizer (e.g., momentum, velocity, step).
        
        Returns:
            tuple: Updated parameters and optimizer state.
        
        The function performs the following steps:
            1. Increments the optimization step counter.
            2. Reshapes the parameters and gradients from tensor form to matrix form.
            3. Computes the Riemannian gradients.
            4. Computes the update direction using momentum and velocity.
            5. Applies bias correction to the learning rate.
            6. Updates the parameters using the Cayley retraction.
            7. Transports the momentum and velocities to the new point on the manifold.
            8. Reshapes the parameters back to tensor form.
        """

        # Compute Riemaniann gradient
        riemannian_grads = self._vec_riemannian_grad(grads, params)

        # Compute update direction given momentum
        mom = tree_map(lambda m, rg: self.beta1 * m + (1 - self.beta1) * rg, 
                       opt_state['momentum'],
                       riemannian_grads)
        
        vel = tree_map(lambda p, rg, v: self.beta2 * v + (1 - self.beta2) * self._metric(p, rg, rg), 
                       params,
                       riemannian_grads,
                       opt_state['velocity'])
        
        # Update rule
        update_direction = tree_map(lambda m, v: m / (jnp.sqrt(v) + self.eps), mom, vel)

        # Bias correction of learning rate
        t = opt_state['iter']
        learning_rate = self.learning_rate * jnp.sqrt(1. - self.beta2 ** t) / (1. - self.beta1 ** t)
        learning_rate = {k: learning_rate for k in params.keys()}

        # Move to new point on manifold via Cayley retraction
        update_direction = tree_map(lambda eta, p: - eta * p, learning_rate, update_direction)
        params = self._vec_cayley_retraction(update_direction, params)

        # Transport moment and velocities to new point on manifold
        opt_state['momentum'] = self._vec_vector_transport(mom, params)
        opt_state['velocity'] = self._vec_vector_transport(vel, params)

        return params, opt_state