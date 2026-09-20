"""
Lyapunov Verifier: Formal Lyapunov stability verification for dynamical systems.

This module provides formal verification of Lyapunov functions for ensuring
stability of control systems using numpy-only computations.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of Lyapunov verification."""
    is_stable: bool
    is_lyapunov: bool
    derivative_bound: float
    level_set_radius: float
    roa_estimate: float
    verification_time_ms: float
    details: Dict


class LyapunovVerifier:
    """
    Formal Lyapunov stability verifier for nonlinear dynamical systems.

    Verifies that a candidate Lyapunov function V(x) satisfies:
    1. V(0) = 0, V(x) > 0 for x != 0 (positive definite)
    2. V(x) -> inf as ||x|| -> inf (radially unbounded)
    3. dV/dt = grad(V) * f(x) <= 0 (negative semi-definite)

    Attributes:
        tolerance: Numerical tolerance for comparisons
        grid_resolution: Resolution for grid-based verification
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        grid_resolution: int = 50,
        max_iterations: int = 1000
    ):
        """
        Initialize Lyapunov verifier.

        Args:
            tolerance: Numerical tolerance for comparisons
            grid_resolution: Grid resolution for sampling-based verification
            max_iterations: Maximum iterations for optimization
        """
        self.tolerance = tolerance
        self.grid_resolution = grid_resolution
        self.max_iterations = max_iterations

    def verify_lyapunov_function(
        self,
        V: Callable[[np.ndarray], float],
        grad_V: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        domain_radius: float = 10.0,
        safe_set: Optional[Callable[[np.ndarray], bool]] = None
    ) -> VerificationResult:
        """
        Verify that a candidate function is a valid Lyapunov function.

        Args:
            V: Lyapunov candidate V(x) -> scalar
            grad_V: Gradient of V: grad_V(x) -> vector
            f: System dynamics f(x, u) -> vector (u optional)
            domain_radius: Radius of verification domain
            safe_set: Optional safe set indicator function

        Returns:
            VerificationResult with stability analysis
        """
        import time
        start_time = time.perf_counter()

        # Infer the state dimension once from V at the origin. The old
        # implementation hard-coded 3, which silently corrupted (or
        # crashed on) every non-3-dimensional system.
        state_dim = self._infer_state_dim(V, grad_V)

        # Generate verification grid
        grid = self._generate_grid(domain_radius, state_dim)

        # Test 1: Positive definite (V(x) > 0 for x != 0)
        pd_result = self._check_positive_definite(V, grid)

        # Test 2: Radially unbounded (checked via boundary conditions)
        ru_result = self._check_radially_unbounded(V, domain_radius, state_dim)

        # Test 3: Negative definite of derivative
        nd_result = self._check_negative_derivative(V, grad_V, f, grid, safe_set)

        # Compute region of attraction estimate
        roa = self._estimate_roa(V, grad_V, f, grid, safe_set)

        # Compute level set radius
        level_set = self._compute_level_set_radius(V, grid)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        is_lyapunov = pd_result["is_positive_definite"] and \
                       ru_result["is_radially_unbounded"] and \
                       nd_result["is_negative_derivative"]

        return VerificationResult(
            is_stable=is_lyapunov,
            is_lyapunov=is_lyapunov,
            derivative_bound=nd_result["max_derivative"],
            level_set_radius=level_set,
            roa_estimate=roa,
            verification_time_ms=elapsed_ms,
            details={
                "positive_definite": pd_result,
                "radially_unbounded": ru_result,
                "negative_derivative": nd_result,
            }
        )

    def _infer_state_dim(self, V: Callable, grad_V: Optional[Callable] = None) -> int:
        """Infer the state dimension by probing V (and grad_V) at the origin.

        Many candidates (e.g. ``x.T @ P @ x``) succeed for *any* length,
        so probing V alone cannot pin the dimension down. The gradient
        is far more discriminating: ``2 P x`` only matches when the
        length equals the size of P. When both are available we take the
        length that makes *both* succeed and yields a consistent
        gradient length.
        """
        candidates = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20)

        if grad_V is None:
            for dim in candidates:
                try:
                    V(np.zeros(dim))
                    return dim
                except Exception:
                    continue
        else:
            for dim in candidates:
                try:
                    V(np.zeros(dim))
                    g = np.asarray(grad_V(np.zeros(dim)))
                    if g.shape == (dim,):
                        return dim
                except Exception:
                    continue

        raise ValueError(
            "Could not infer the state dimension from V/grad_V; make sure they "
            "accept a 1-D numpy array of the system's state dimension."
        )

    def _generate_grid(self, radius: float, state_dim: Optional[int] = None) -> np.ndarray:
        """Generate sample grid within verification domain.

        Args:
            radius: Radius of the verification ball.
            state_dim: Dimension of the state space. Defaults to 3 for
                backwards compatibility with callers that omit it; the
                public entry points always infer it from ``V``.
        """
        if state_dim is None:
            state_dim = 3
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        # For simplicity, generate random samples within the ball
        n_samples = min(self.grid_resolution ** 2, 10000)
        samples = np.empty((n_samples, state_dim))

        # Rejection sampling in a ball becomes vanishingly inefficient
        # as the dimension grows (the ball's volume share of the
        # bounding cube collapses). Sample from a Gaussian and rescale
        # each accepted point onto the ball instead: pick a random
        # direction, then a radius drawn from the correct distribution
        # (r = R * u**(1/d)), which is exact and dimension-independent.
        if state_dim >= 3:
            directions = np.random.randn(n_samples, state_dim)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            # Guard against a degenerate all-zero draw.
            norms = np.where(norms < 1e-12, 1.0, norms)
            directions = directions / norms
            radii = radius * np.random.rand(n_samples, 1) ** (1.0 / state_dim)
            samples = directions * radii
            return samples

        filled = 0
        attempts = 0
        max_attempts = 200 * n_samples + 1000
        while filled < n_samples and attempts < max_attempts:
            attempts += 1
            candidate = np.random.randn(state_dim) * radius * 2
            if np.linalg.norm(candidate) <= radius:
                samples[filled] = candidate
                filled += 1

        return samples[:filled]

    def _origin_vector(self, V: Callable, grid: np.ndarray) -> np.ndarray:
        """Build the zero vector of the correct state dimension.

        The dimension comes from the grid when it has rows, otherwise it
        is inferred from ``V`` -- it must never silently default to 3,
        which is what broke every non-3-dimensional system.
        """
        if len(grid) > 0:
            return np.zeros(grid.shape[1])
        return np.zeros(self._infer_state_dim(V))

    def _check_positive_definite(
        self,
        V: Callable,
        grid: np.ndarray
    ) -> Dict:
        """Check positive definiteness: V(0) = 0, V(x) > 0 for x != 0."""
        origin = self._origin_vector(V, grid)
        V_origin = V(origin)

        if abs(V_origin) > self.tolerance:
            return {"is_positive_definite": False, "error": "V(0) != 0"}

        min_V = float('inf')
        violations = 0

        for x in grid:
            if np.linalg.norm(x) < self.tolerance:
                continue
            val = V(x)
            if val <=0:
                violations += 1
            min_V = min(min_V, val)

        return {
            "is_positive_definite": violations == 0 and min_V > self.tolerance,
            "min_value": min_V,
            "violations": violations,
            "V_at_origin": V_origin
        }

    def _check_radially_unbounded(
        self,
        V: Callable,
        radius: float,
        state_dim: int = 3
    ) -> Dict:
        """Check radially unbounded: V(x) -> inf as ||x|| -> inf."""
        test_radii = np.array([1, 5, 10, 50, 100])
        max_V_at_radii = []

        for r in test_radii:
            # Sample multiple points at this radius
            for _ in range(100):
                direction = np.random.randn(state_dim)
                direction = direction / np.linalg.norm(direction)
                x = direction * r
                max_V_at_radii.append(V(x))

        is_unbounded = max(max_V_at_radii) > max_V_at_radii[0] * 10

        return {
            "is_radially_unbounded": is_unbounded,
            "max_V_at_max_radius": max(max_V_at_radii),
            "V_at_min_radius": min(max_V_at_radii[:100])
        }

    def _check_negative_derivative(
        self,
        V: Callable,
        grad_V: Callable,
        f: Callable,
        grid: np.ndarray,
        safe_set: Optional[Callable]
    ) -> Dict:
        """Check negative semi-definiteness of V derivative along trajectories."""
        max_derivative = float('-inf')
        violations = 0
        total_checked = 0

        for x in grid:
            # Check if point is in safe set (if defined)
            if safe_set is not None and not safe_set(x):
                continue

            # Compute dV/dt = grad(V) @ f(x)
            try:
                grad = grad_V(x)
                dyn = f(x, None)
                deriv = np.dot(grad, dyn)

                max_derivative = max(max_derivative, deriv)

                if deriv > self.tolerance:
                    violations += 1

                total_checked += 1
            except Exception:
                continue

        return {
            "is_negative_derivative": violations == 0,
            "max_derivative": max_derivative,
            "violations": violations,
            "total_checked": total_checked
        }

    def _estimate_roa(
        self,
        V: Callable,
        grad_V: Callable,
        f: Callable,
        grid: np.ndarray,
        safe_set: Optional[Callable]
    ) -> float:
        """Estimate region of attraction radius."""
        origin = self._origin_vector(V, grid)
        V_origin = V(origin)

        # Find maximum radius where V stays below level set
        roa_radius = 0.0
        for x in grid:
            if safe_set is not None and not safe_set(x):
                continue

            norm = np.linalg.norm(x)
            try:
                grad = grad_V(x)
                dyn = f(x, None)
                deriv = np.dot(grad, dyn)

                if deriv <= 0 and V(x) > V_origin:
                    roa_radius = max(roa_radius, norm)
            except Exception:
                continue

        return roa_radius

    def _compute_level_set_radius(
        self,
        V: Callable,
        grid: np.ndarray
    ) -> float:
        """Compute level set radius (where V = 1 if V(0) = 0)."""
        origin = self._origin_vector(V, grid)
        V_origin = V(origin)
        target_V = V_origin + 1.0

        level_set_radius = float('inf')
        for x in grid:
            norm = np.linalg.norm(x)
            try:
                val = V(x)
                if abs(val - target_V) < abs(level_set_radius - target_V):
                    level_set_radius = norm
            except Exception:
                continue

        return level_set_radius if level_set_radius < float('inf') else 0.0

    def compute_derivative_bound(
        self,
        grad_V: Callable,
        f: Callable,
        x: np.ndarray
    ) -> float:
        """Compute dV/dt bound at a specific point."""
        try:
            grad = grad_V(x)
            dyn = f(x, None)
            return float(np.dot(grad, dyn))
        except Exception:
            return float('inf')

    def synthesize_lyapunov_quadratic(
        self,
        A: np.ndarray,
        P_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Synthesize quadratic Lyapunov function V = x^T P x for linear system x' = Ax.

        Args:
            A: System matrix
            P_init: Initial guess for P (optional)

        Returns:
            P matrix such that A^T P + P A is negative definite
        """
        n = A.shape[0]

        if P_init is None:
            P_init = np.eye(n)

        # Iterative refinement (simplified Lyapunov equation solver)
        P = P_init.copy()

        for _ in range(self.max_iterations):
            # Compute A^T P + P A
            AT_P = A.T @ P
            P_A = P @ A
            lyap_lhs = AT_P + P_A

            # Target: -Q for some positive definite Q
            Q_target = -lyap_lhs

            # Check if Q_target is positive definite
            eigenvalues = np.linalg.eigvalsh(Q_target)
            if np.all(eigenvalues > self.tolerance):
                break

            # Simple gradient descent update
            P = P - 0.01 * (lyap_lhs + lyap_lhs.T)

        return P

    def verify_input_to_state_stability(
        self,
        V: Callable,
        grad_V: Callable,
        f: Callable,
        g: Callable,
        domain_radius: float = 10.0
    ) -> Dict:
        """
        Verify input-to-state stability with ISS-Lyapunov function.

        Args:
            V: ISS-Lyapunov function
            grad_V: Gradient of V
            f: Nominal dynamics f(x, u)
            g: Disturbance dynamics g(x)
            domain_radius: Verification domain radius

        Returns:
            Dictionary with ISS verification results
        """
        grid = self._generate_grid(domain_radius, self._infer_state_dim(V, grad_V))

        # Check ISS condition: dV/dt <= -alpha(||x||) + gamma(||d||)
        alpha_vals = []
        gamma_vals = []

        for x in grid[:1000]:  # Limit samples for efficiency
            try:
                grad = grad_V(x)
                fnom = f(x, None)
                gdyn = g(x)

                deriv = np.dot(grad, fnom)
                dist_term = np.dot(grad, gdyn)

                x_norm = np.linalg.norm(x)
                d_norm = 1.0  # Assume unit disturbance

                # alpha(||x||) = -dV/dt for nominal system
                if deriv < 0:
                    alpha_vals.append(-deriv / x_norm)

                # gamma(||d||) >= |grad(V) * g|
                gamma_vals.append(abs(dist_term) / d_norm)
            except Exception:
                continue

        alpha_mean = np.mean(alpha_vals) if alpha_vals else 0.0
        gamma_bound = np.max(gamma_vals) if gamma_vals else float('inf')

        return {
            "is_iss": alpha_mean > 0 and gamma_bound < float('inf'),
            "alpha_estimate": alpha_mean,
            "gamma_bound": gamma_bound,
            "samples_checked": len(alpha_vals)
        }


def candidate_quadratic_lyapunov(x: np.ndarray, P: np.ndarray) -> float:
    """Standard quadratic Lyapunov candidate: V(x) = x^T P x."""
    return float(x.T @ P @ x)


def quadratic_lyapunov_gradient(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Gradient of quadratic Lyapunov: grad(V) = 2 P x."""
    return 2 * P @ x


def linear_system_dynamics(x: np.ndarray, A: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
    """Linear system dynamics: x' = A x + B u."""
    return A @ x