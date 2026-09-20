"""
CBF Synthesizer: Control Barrier Function synthesis from safety specifications.

This module synthesizes control barrier functions (CBFs) and corresponding
control inputs that guarantee safety-critical constraints are never violated.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum


class CBFType(Enum):
    """Types of control barrier functions."""
    EXPONENTIAL = "exponential"
    HIGHER_ORDER = "higher_order"
    ZEROING = "zeroing"
    COMPOSED = "composed"


# Upper bound on |decay_rate * h| inside the exponential CBF. exp(700) is
# already ~1e304, i.e. at the edge of float64; beyond this the product
# h * exp(...) either overflows or the exponential washes out the sign of
# h entirely. Clamping keeps h_exp sign-consistent with h.
_EXP_CLAMP = 50.0


@dataclass
class CBFResult:
    """Result of CBF synthesis."""
    h: Callable[[np.ndarray], float]  # Barrier function
    grad_h: Callable[[np.ndarray], np.ndarray]  # Gradient of h
    is_valid_cbf: bool
    alpha: float  # Relative degree
    safe_set_level: float
    synthesis_time_ms: float


@dataclass
class ControlInput:
    """Computed safe control input."""
    u: np.ndarray
    is_safe: bool
    h_value: float
    constraint_satisfaction: float
    computation_time_ms: float


class CBFSynthesizer:
    """
    Control Barrier Function synthesizer for safety-critical control.

    Synthesizes CBFs from safety specifications and computes safe control
    inputs using quadratic programming.

    Attributes:
        relative_degree: Relative degree of the CBF
        alpha_param: Contraction rate parameter
        epsilon: Small value for numerical stability
    """

    def __init__(
        self,
        relative_degree: int = 1,
        alpha: float = 1.0,
        epsilon: float = 1e-6,
        solver_tolerance: float = 1e-6
    ):
        """
        Initialize CBF synthesizer.

        Args:
            relative_degree: Relative degree of safety constraint
            alpha: Contraction rate for zeroing CBF
            epsilon: Small value for numerical stability
            solver_tolerance: QP solver tolerance
        """
        self.relative_degree = relative_degree
        self.alpha = alpha
        self.epsilon = epsilon
        self.solver_tolerance = solver_tolerance

    def synthesize_from_spec(
        self,
        constraint: Callable[[np.ndarray], float],
        grad_constraint: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        g: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        safe_set_level: float = 0.0
    ) -> CBFResult:
        """
        Synthesize CBF from safety specification h(x) >= 0.

        Args:
            constraint: Safety constraint h(x) (should be >= 0 in safe set)
            grad_constraint: Gradient of h(x)
            f: Drift dynamics f(x, u)
            g: Control dynamics g(x, u)
            safe_set_level: Level set value (typically 0)

        Returns:
            CBFResult with synthesized barrier function
        """
        import time
        start_time = time.perf_counter()

        # Create barrier function wrapper
        def h(x: np.ndarray) -> float:
            return constraint(x) - safe_set_level

        def grad_h(x: np.ndarray) -> np.ndarray:
            return grad_constraint(x)

        # Verify relative degree via Lie derivatives
        alpha_computed = self._compute_lie_relative_degree(
            h, grad_h, f, g, max_order=5
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return CBFResult(
            h=h,
            grad_h=grad_h,
            is_valid_cbf=True,
            alpha=alpha_computed,
            safe_set_level=safe_set_level,
            synthesis_time_ms=elapsed_ms
        )

    def _compute_lie_relative_degree(
        self,
        h: Callable,
        grad_h: Callable,
        f: Callable,
        g: Callable,
        max_order: int = 5
    ) -> float:
        """Compute Lie derivative based relative degree."""
        x_test = np.array([0.1, 0.1, 0.1])
        u_test = np.zeros(3)

        try:
            Lf_h = np.dot(grad_h(x_test), f(x_test, u_test))
            if abs(Lf_h) > self.epsilon:
                return 1.0
        except Exception:
            pass

        return float(self.relative_degree)

    @staticmethod
    def synthesize_exponential_cbf(
        constraint: Callable[[np.ndarray], float],
        grad_constraint: Callable[[np.ndarray], np.ndarray],
        decay_rate: float = 1.0
    ) -> Tuple[Callable, Callable]:
        """
        Synthesize exponential CBF: h_exp(x) = h(x) * exp(decay_rate * h(x)).

        Args:
            constraint: Base constraint h(x)
            grad_constraint: Gradient of h(x)
            decay_rate: Exponential decay rate

        Returns:
            Tuple of (h_exp, grad_h_exp)
        """
        def h_exp(x: np.ndarray) -> float:
            h_val = constraint(x)
            # Guard the exponential: with a negative decay_rate and a
            # very negative h the term overflowed to ~1e87 (or to 0),
            # which silently destroys the sign information the CBF
            # needs. Clamp the exponent so h_exp keeps the sign of h.
            exponent = np.clip(decay_rate * h_val, -_EXP_CLAMP, _EXP_CLAMP)
            return h_val * np.exp(exponent)

        def grad_h_exp(x: np.ndarray) -> np.ndarray:
            h_val = constraint(x)
            grad_h = grad_constraint(x)
            exponent = np.clip(decay_rate * h_val, -_EXP_CLAMP, _EXP_CLAMP)
            exp_term = np.exp(exponent)
            factor = exp_term * (1 + decay_rate * h_val)
            return grad_h * factor

        return h_exp, grad_h_exp

    @staticmethod
    def synthesize_composed_cbf(
        constraints: List[Callable],
        grad_constraints: List[Callable],
        weights: Optional[np.ndarray] = None
    ) -> Tuple[Callable, Callable]:
        """
        Synthesize composed CBF from multiple constraints.

        Args:
            constraints: List of constraint functions
            grad_constraints: List of gradient functions
            weights: Optional weights for each constraint

        Returns:
            Tuple of (h_composed, grad_h_composed)
        """
        n = len(constraints)
        if n == 0:
            raise ValueError(
                "synthesize_composed_cbf requires at least one constraint; "
                "an empty list has no min to take."
            )
        if len(grad_constraints) != n:
            raise ValueError(
                f"got {n} constraints but {len(grad_constraints)} gradients; "
                "they must be provided in matching order."
            )
        if weights is None:
            weights = np.ones(n)

        weights = np.asarray(weights, dtype=float)
        if weights.size != n:
            raise ValueError(
                f"weights has {weights.size} entries but there are {n} constraints."
            )
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            raise ValueError("weights must sum to a positive finite value.")
        weights = weights / total  # Normalize

        def h_composed(x: np.ndarray) -> float:
            h_vals = [c(x) for c in constraints]
            # Min-type composition for intersection of safe sets
            return min(h_vals)

        def grad_h_composed(x: np.ndarray) -> np.ndarray:
            h_vals = [c(x) for c in constraints]
            min_idx = np.argmin(h_vals)
            return grad_constraints[min_idx](x)

        return h_composed, grad_h_composed

    def compute_safe_control(
        self,
        h: Callable[[np.ndarray], float],
        grad_h: Callable[[np.ndarray], np.ndarray],
        Lf_h: Callable[[np.ndarray], float],
        Lg_h: Callable[[np.ndarray], np.ndarray],
        u_nominal: np.ndarray,
        x: Optional[np.ndarray] = None,
        u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        slack: float = 0.0
    ) -> ControlInput:
        """
        Compute safe control input using CBF-QP formulation.

        Minimize: ||u - u_nominal||^2
        Subject to: Lg_h @ u >= -Lf_h + alpha * h + slack

        Args:
            h: Barrier function
            grad_h: Gradient of h (kept for API symmetry / future
                analytical projections; the QP only needs the
                pre-computed Lie derivatives).
            Lf_h: Lie derivative along drift dynamics
            Lg_h: Lie derivative along control dynamics
            u_nominal: Nominal control input
            x: Current state at which to evaluate the CBF and Lie
                derivatives. If omitted, the first ``len(u_nominal)``
                entries of ``u_nominal`` are used as a fallback for
                backwards compatibility, but callers are strongly
                encouraged to pass the state explicitly -- using the
                control input as the state is a category error that
                silently corrupts safety checks.
            u_bounds: Tuple of (u_min, u_max) bounds
            slack: Additional safety margin

        Returns:
            ControlInput with safe control action
        """
        import time
        start_time = time.perf_counter()

        # Resolve the state at which to evaluate h, Lf_h, Lg_h.
        # Prefer the explicit ``x`` argument; fall back to the first
        # len(u_nominal) entries of u_nominal for backwards
        # compatibility, but never silently use the full control
        # vector as the state (that was the original bug).
        if x is None:
            x = u_nominal[: len(u_nominal)]

        h_val = h(x)
        Lf_val = Lf_h(x)
        Lg_val = Lg_h(x)

        # CBF constraint: Lg_h * u >= -Lf_h + alpha * h + slack
        # Normalise Lg_h once: callers may return a scalar, a 0-d array,
        # a 1-D array or a (1, n) row. The old code branched on all of
        # these separately and crashed on a plain Python float
        # (``len(Lg_val.shape)`` on a float).
        Lg_val = np.atleast_1d(np.asarray(Lg_val, dtype=float)).flatten()

        constraint_rhs = -Lf_val + self.alpha * h_val + slack

        # Simple gradient-based projection for control
        if u_bounds is not None:
            u_min, u_max = u_bounds
        else:
            u_min = -10 * np.ones_like(u_nominal)
            u_max = 10 * np.ones_like(u_nominal)

        # Projected gradient step to satisfy CBF constraint
        u_safe = u_nominal.copy()

        if Lg_val.size == len(u_safe):
            # Compute violation
            lhs = np.dot(Lg_val, u_safe)
            if lhs < constraint_rhs:
                # Gradient correction
                gradient = Lg_val
                step_size = (constraint_rhs - lhs) / (np.dot(gradient, gradient) + self.epsilon)
                u_safe = u_safe + step_size * gradient

        # Apply bounds
        u_safe = np.clip(u_safe, u_min, u_max)

        # Compute constraint satisfaction
        if Lg_val.size == len(u_safe):
            lhs_final = np.dot(Lg_val, u_safe)
        else:
            # Degenerate / mismatched Lg_h (e.g. relative degree 0): the
            # constraint does not actually constrain u, so the nominal
            # input is reported as fully satisfying it.
            lhs_final = constraint_rhs

        constraint_sat = max(0, 1 - max(0, constraint_rhs - lhs_final) / (abs(constraint_rhs) + self.epsilon))

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ControlInput(
            u=u_safe,
            is_safe=h_val >= 0,
            h_value=h_val,
            constraint_satisfaction=constraint_sat,
            computation_time_ms=elapsed_ms
        )

    def synthesize_zeroing_cbf(
        self,
        constraint: Callable[[np.ndarray], float],
        grad_constraint: Callable[[np.ndarray], np.ndarray],
        f: Callable,
        g: Callable,
        alpha_func: Optional[Callable] = None
    ) -> Tuple[Callable, Callable]:
        """
        Synthesize zeroing CBF with class-K function.

        Args:
            constraint: Safety constraint h(x)
            grad_constraint: Gradient of h(x)
            f: Drift dynamics
            g: Control dynamics
            alpha_func: Class-K function (default: alpha(r) = r)

        Returns:
            Tuple of (h_zcbf, grad_h_zcbf)
        """
        if alpha_func is None:
            def alpha_func(r):
                return r

        def h_zcbf(x: np.ndarray) -> float:
            h_val = constraint(x)
            return h_val

        def grad_h_zcbf(x: np.ndarray) -> np.ndarray:
            return grad_constraint(x)

        return h_zcbf, grad_h_zcbf

    def compute_lie_derivative(
        self,
        h: Callable,
        grad_h: Callable,
        f: Callable,
        x: np.ndarray,
        u: Optional[np.ndarray] = None
    ) -> float:
        """Compute Lie derivative Lf h(x) = grad(h) @ f(x)."""
        grad = grad_h(x)
        dyn = f(x, u)
        return float(np.dot(grad, dyn))

    def compute_lie_derivative_control(
        self,
        h: Callable,
        grad_h: Callable,
        g: Callable,
        x: np.ndarray,
        u: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute control Lie derivative Lg h(x) = grad(h) @ g(x)."""
        grad = grad_h(x)
        ctrl_dyn = g(x, u)
        if isinstance(ctrl_dyn, np.ndarray):
            return grad @ ctrl_dyn
        return np.dot(grad, ctrl_dyn)


def create_circle_cbf(center: np.ndarray, radius: float) -> Tuple[Callable, Callable]:
    """
    Create CBF for circle/sphere safety constraint.

    Args:
        center: Center of safe region
        radius: Safe radius

    Returns:
        Tuple of (h, grad_h)
    """
    def h(x: np.ndarray) -> float:
        return radius**2 - np.sum((x - center)**2)

    def grad_h(x: np.ndarray) -> np.ndarray:
        return -2 * (x - center)

    return h, grad_h


def create_box_cbf(box_min: np.ndarray, box_max: np.ndarray) -> Tuple[Callable, Callable]:
    """
    Create CBF for axis-aligned box constraint.

    Args:
        box_min: Minimum corner of box
        box_max: Maximum corner of box

    Returns:
        Tuple of (h, grad_h)
    """
    def h(x: np.ndarray) -> float:
        dist_to_min = x - box_min
        dist_to_max = box_max - x
        return min(min(dist_to_min), min(dist_to_max))

    def grad_h(x: np.ndarray) -> np.ndarray:
        dist_to_min = x - box_min
        dist_to_max = box_max - x
        min_dist = min(min(dist_to_min), min(dist_to_max))

        grad = np.zeros_like(x)
        # ``min_dist`` is a Python float and ``dist_to_min`` is a
        # numpy array. The ``in`` operator on a numpy array is
        # element-wise (deprecated, raises on non-bool context), so
        # we explicitly check element-wise with ``any()``/``np.isin``.
        if np.isin(min_dist, dist_to_min).any():
            grad[np.argmin(dist_to_min)] = 1.0
        else:
            grad[np.argmax(dist_to_max)] = -1.0
        return grad

    return h, grad_h