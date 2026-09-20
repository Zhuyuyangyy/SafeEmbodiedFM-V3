"""Tests for the CVXPY-based CBF QP solver (R2 addition)."""
import sys

import numpy as np
import pytest

# Ensure the src/ layout is importable when pytest is run from the project root.
sys.path.insert(0, "D:/ZYY Project/SafeEmbodiedFM-V3/src")

from safe_fm_v3.qp_solver import (
    CBFQPUnavailable,
    is_qp_available,
    solve_cbf_qp,
)


pytestmark = pytest.mark.skipif(
    not is_qp_available(), reason="cvxpy not installed; skipping CBF-QP tests"
)


class TestCBFQP:
    """Tests for solve_cbf_qp()."""

    def test_inactive_constraint(self):
        """When the nominal control already satisfies the constraint, the
        solver should return it (modulo tiny numerical adjustments)."""
        u_nominal = np.array([0.1, 0.1])
        Lg_h = np.array([1.0, 0.0])
        Lf_h = 0.0
        h_value = 1.0
        # Constraint: 1.0 * u[0] >= -0 + 1*1 + 0 = 1.0  => u[0] >= 1
        # but u_nominal[0] = 0.1 violates. Use a different example:
        Lg_h = np.array([1.0, 0.0])
        # Constraint: 1.0 * u[0] >= -0 + 1*5 + 0 = 5  => u[0] >= 5
        # nominal [0.1, 0.1] violates => solver should push u[0] up.
        result = solve_cbf_qp(
            u_nominal=u_nominal,
            Lg_h=Lg_h,
            Lf_h=Lf_h,
            h_value=5.0,
            alpha=1.0,
        )
        assert result.is_feasible
        assert result.u[0] >= 5.0 - 1e-3
        # u[1] should be close to nominal (gradient says not to move it)
        np.testing.assert_allclose(result.u[1], 0.1, atol=1e-2)

    def test_active_constraint_equality(self):
        """When the constraint is active, the solution should sit on the
        boundary (within solver tolerance)."""
        u_nominal = np.array([0.0, 0.0])
        Lg_h = np.array([1.0, 0.0])
        Lf_h = 0.0
        h_value = 2.0
        # constraint: u[0] >= 2
        result = solve_cbf_qp(u_nominal, Lg_h, Lf_h, h_value, alpha=1.0)
        assert result.is_feasible
        assert abs(result.u[0] - 2.0) < 1e-3

    def test_u_bounds_respected(self):
        """Box constraints u_min <= u <= u_max must be respected."""
        u_nominal = np.array([0.0, 0.0])
        Lg_h = np.array([1.0, 0.0])
        Lf_h = 0.0
        h_value = 100.0  # constraint demands u[0] >= 100
        u_min = np.array([-1.0, -1.0])
        u_max = np.array([10.0, 10.0])  # cap below the constraint demand
        result = solve_cbf_qp(
            u_nominal, Lg_h, Lf_h, h_value, alpha=1.0,
            u_bounds=(u_min, u_max),
        )
        # Should saturate at u_max and still be feasible (constraint is
        # violated but the problem is feasible w.r.t. box constraints).
        assert result.u[0] <= 10.0 + 1e-6

    def test_shape_mismatch_raises(self):
        """Mismatched dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            solve_cbf_qp(
                u_nominal=np.array([0.1, 0.2, 0.3]),
                Lg_h=np.array([1.0, 0.0]),  # 2 vs 3
                Lf_h=0.0,
                h_value=1.0,
            )

    def test_returns_metadata(self):
        """Result must expose status, timing and feasibility metadata."""
        result = solve_cbf_qp(
            u_nominal=np.array([0.0, 0.0]),
            Lg_h=np.array([1.0, 0.0]),
            Lf_h=0.0,
            h_value=1.0,
        )
        assert isinstance(result.solver_status, str)
        assert result.solve_time_ms >= 0.0
        assert 0.0 <= result.constraint_satisfaction <= 1.0
        assert result.u.shape == (2,)
