"""
CVXPY-based Quadratic Programming (QP) solver for Control Barrier Functions.

This module provides a true QP solver for the CBF safety filter
formulation used throughout SafeEmbodiedFM-V3:

    minimise    (1/2) * ||u - u_nominal||^2
    subject to  Lg_h(x) @ u  >=  -Lf_h(x) + alpha * h(x) + slack
                u_min <= u <= u_max        (optional)

The default CBF solver in :mod:`safe_fm_v3.cbf_synthesizer` uses a
projected-gradient step which is fast but approximate. This module
provides a numerically-robust reference implementation using CVXPY
that is intended for offline analysis, hardware-in-the-loop testing
and as ground-truth for comparing the projected-gradient approximation.

The module is decoupled from the rest of the package -- importing it
does not require CVXPY at module load time. If CVXPY is not installed,
:class:`CBFQPUnavailable` is raised only when the solver is invoked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # CVXPY is an optional dependency for the QP solver
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:  # pragma: no cover - import-time guard
    HAS_CVXPY = False


class CBFQPUnavailable(RuntimeError):
    """Raised when the CVXPY-based QP solver is requested but CVXPY is
    not importable in the current environment."""


@dataclass
class CBFQPResult:
    """Result of a CBF-QP solve.

    Attributes:
        u: Optimal safe control input.
        is_feasible: Whether CVXPY found a feasible (and optimal) solution.
        h_value: Value of the barrier function at the state.
        constraint_satisfaction: 1.0 if constraint active and satisfied,
            decreases linearly as the constraint is violated.
        solver_status: CVXPY status string (``"optimal"``,
            ``"infeasible"``, ``"unbounded"`` ...).
        solve_time_ms: Wall-clock solve time in milliseconds.
        slack_used: Additional safety margin that was added.
    """

    u: np.ndarray
    is_feasible: bool
    h_value: float
    constraint_satisfaction: float
    solver_status: str
    solve_time_ms: float
    slack_used: float


def solve_cbf_qp(
    u_nominal: np.ndarray,
    Lg_h: np.ndarray,
    Lf_h: float,
    h_value: float,
    alpha: float = 1.0,
    slack: float = 0.0,
    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    solver: str = "OSQP",
    eps: float = 1e-6,
) -> CBFQPResult:
    """Solve the CBF safety-filter QP.

    Args:
        u_nominal: Nominal (unsafe) control input, shape ``(n,)``.
        Lg_h: Lie derivative of ``h`` along control dynamics,
            shape ``(n,)``. May be 1-D or 2-D ``(1, n)``; will be
            flattened.
        Lf_h: Lie derivative of ``h`` along drift dynamics, scalar.
        h_value: Value of the barrier function at the current state.
        alpha: Class-K rate for the zeroing CBF.
        slack: Additional safety margin (>= 0).
        u_bounds: Optional ``(u_min, u_max)`` element-wise box constraints.
        solver: CVXPY backend to use (``"OSQP"``, ``"CLARABEL"``, ``"SCS"`` ...).
        eps: Small regularisation added to the cost Hessian for numerical
            robustness.

    Returns:
        :class:`CBFQPResult` with the optimal control and solver metadata.

    Raises:
        CBFQPUnavailable: If CVXPY is not importable.
    """
    if not HAS_CVXPY:
        raise CBFQPUnavailable(
            "CVXPY is not installed. Install with `pip install cvxpy` to use "
            "solve_cbf_qp()."
        )

    u_nominal = np.asarray(u_nominal, dtype=np.float64).flatten()
    n = u_nominal.size
    Lg_h = np.asarray(Lg_h, dtype=np.float64).flatten()
    if Lg_h.size != n:
        raise ValueError(
            f"Lg_h has {Lg_h.size} elements but u_nominal has {n}; "
            "the dimensions must match for the CBF constraint."
        )

    rhs = -float(Lf_h) + alpha * float(h_value) + float(slack)

    u = cp.Variable(n)
    cost = 0.5 * cp.sum_squares(u - u_nominal) + eps * cp.sum_squares(u)
    constraints = [Lg_h @ u >= rhs]
    if u_bounds is not None:
        u_min, u_max = u_bounds
        u_min = np.asarray(u_min, dtype=np.float64).flatten()
        u_max = np.asarray(u_max, dtype=np.float64).flatten()
        if u_min.size != n or u_max.size != n:
            raise ValueError("u_bounds must have the same length as u_nominal")
        constraints.append(u >= u_min)
        constraints.append(u <= u_max)

    prob = cp.Problem(cp.Minimize(cost), constraints)

    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception as exc:  # pragma: no cover - cvxpy-specific failure
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return CBFQPResult(
            u=u_nominal,
            is_feasible=False,
            h_value=float(h_value),
            constraint_satisfaction=0.0,
            solver_status=f"error:{exc}",
            solve_time_ms=elapsed_ms,
            slack_used=float(slack),
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    is_feasible = prob.status in ("optimal", "optimal_inaccurate")
    if is_feasible and u.value is not None:
        u_opt = np.asarray(u.value, dtype=np.float64).flatten()
    else:
        u_opt = u_nominal

    if is_feasible:
        lhs_final = float(Lg_h @ u_opt)
        denom = abs(rhs) + 1e-9
        sat = max(0.0, min(1.0, 1.0 - max(0.0, rhs - lhs_final) / denom))
    else:
        sat = 0.0

    return CBFQPResult(
        u=u_opt,
        is_feasible=is_feasible,
        h_value=float(h_value),
        constraint_satisfaction=sat,
        solver_status=str(prob.status),
        solve_time_ms=elapsed_ms,
        slack_used=float(slack),
    )


def is_qp_available() -> bool:
    """Return True if the CVXPY backend can be used in this environment."""
    return HAS_CVXPY
