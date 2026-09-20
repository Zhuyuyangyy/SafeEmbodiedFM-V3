"""
Test suite for SafeEmbodiedFM-V3.

This module contains comprehensive tests for all safety verification
and control components.
"""

import numpy as np
import pytest
import time
import warnings
from typing import Tuple, Callable, Optional

# Import modules under test
import sys
sys.path.insert(0, 'D:/ZYY Project/SafeEmbodiedFM-V3/src')

from safe_fm_v3 import (
    LyapunovVerifier,
    CBFSynthesizer,
    RuntimeMonitor,
    SafePlanner,
    CollisionDetector,
    SafetyBounds,
    PlannerConfig,
    PlannerType
)


# ============================================================================
# LYAPUNOV VERIFIER TESTS
# ============================================================================

class TestLyapunovVerifier:
    """Tests for Lyapunov stability verification."""

    def test_quadratic_lyapunov_candidate(self):
        """Test standard quadratic Lyapunov function."""
        P = np.eye(3)
        x = np.array([0.5, 0.3, 0.1])

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x  # Stable linear system

        verifier = LyapunovVerifier(grid_resolution=30)
        result = verifier.verify_lyapunov_function(V, grad_V, f, domain_radius=5.0)

        assert result.is_lyapunov, "Quadratic Lyapunov for stable system should verify"
        assert result.derivative_bound <= 0, "Derivative should be non-positive"

    def test_unstable_system_lyapunov(self):
        """Test that Lyapunov fails for unstable system."""
        P = np.eye(3)
        x = np.array([0.5, 0.3, 0.1])

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return x  # Unstable system x' = x

        verifier = LyapunovVerifier(grid_resolution=30)
        result = verifier.verify_lyapunov_function(V, grad_V, f, domain_radius=5.0)

        assert not result.is_lyapunov or result.derivative_bound > 0

    def test_positive_definiteness(self):
        """Test positive definiteness check."""
        P = np.eye(3)

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x

        verifier = LyapunovVerifier()
        grid = verifier._generate_grid(10.0)

        result = verifier._check_positive_definite(V, grid)
        assert result["is_positive_definite"], "V(x) = x^T P x should be positive definite"
        assert result["V_at_origin"] == pytest.approx(0.0, abs=1e-5)

    def test_radially_unbounded(self):
        """Test radially unbounded check."""
        P = np.eye(3)

        def V(x): return float(x.T @ P @ x)

        verifier = LyapunovVerifier()
        result = verifier._check_radially_unbounded(V, 100.0)

        assert result["is_radially_unbounded"], "Quadratic should be radially unbounded"

    def test_linear_system_lyapunov(self):
        """Test Lyapunov synthesis for linear system."""
        # Stable linear system: x' = -x
        A = -np.eye(3)

        verifier = LyapunovVerifier()
        P = verifier.synthesize_lyapunov_quadratic(A)

        assert P.shape == (3, 3), "P should be 3x3"
        assert np.allclose(P, P.T), "P should be symmetric"

        # Verify Lyapunov equation
        lyap_product = A.T @ P + P @ A
        eigenvalues = np.linalg.eigvalsh(lyap_product)
        assert np.all(eigenvalues < 0), "Lyapunov equation should be satisfied"

    def test_derivative_computation(self):
        """Test derivative bound computation."""
        P = np.eye(3)
        x = np.array([1.0, 0.5, 0.2])

        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x

        verifier = LyapunovVerifier()
        deriv = verifier.compute_derivative_bound(grad_V, f, x)

        expected = float(np.dot(2 * P @ x, -x))
        assert abs(deriv - expected) < 1e-10

    def test_input_to_state_stability(self):
        """Test ISS-Lyapunov verification."""
        def V(x): return float(x.T @ np.eye(3) @ x)
        def grad_V(x): return 2 * x
        def f(x, u): return -x
        def g(x): return x * 0.1  # Disturbance term

        verifier = LyapunovVerifier(grid_resolution=20)
        result = verifier.verify_input_to_state_stability(V, grad_V, f, g)

        assert "is_iss" in result
        assert "alpha_estimate" in result
        assert "gamma_bound" in result

    def test_roa_estimation(self):
        """Test region of attraction estimation."""
        P = np.eye(3)

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x

        verifier = LyapunovVerifier(grid_resolution=30)
        grid = verifier._generate_grid(10.0)

        roa = verifier._estimate_roa(V, grad_V, f, grid, None)
        assert roa >= 0, "ROA should be non-negative"


# ============================================================================
# CBF SYNTHESIZER TESTS
# ============================================================================

class TestCBFSynthesizer:
    """Tests for Control Barrier Function synthesis."""

    def test_circle_cbf_creation(self):
        """Test circle/sphere CBF creation."""
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0

        h, grad_h = CBFSynthesizer.synthesize_exponential_cbf(
            lambda x: radius**2 - np.sum((x - center)**2),
            lambda x: -2 * (x - center),
            decay_rate=1.0
        )

        # Test inside safe set
        inside_point = np.array([0.0, 0.0, 0.0])
        assert h(inside_point) > 0, "Origin should be inside safe set"

        # Test on boundary
        boundary_point = np.array([1.0, 0.0, 0.0])
        assert abs(h(boundary_point)) < 1e-10

        # Test gradient
        grad = grad_h(boundary_point)
        assert np.allclose(grad, -2 * (boundary_point - center))

    def test_cbf_from_spec(self):
        """Test CBF synthesis from specification."""
        def constraint(x): return 1.0 - np.linalg.norm(x)
        def grad_constraint(x): return -x / (np.linalg.norm(x) + 1e-6)
        def f(x, u): return -x
        def g(x, u): return np.eye(3)

        synthesizer = CBFSynthesizer()
        result = synthesizer.synthesize_from_spec(
            constraint, grad_constraint, f, g
        )

        assert result.is_valid_cbf
        assert result.alpha >= 1.0

    def test_safe_control_computation(self):
        """Test safe control input computation."""
        def h(x): return 1.0 - np.linalg.norm(x)
        def grad_h(x): return -x / (np.linalg.norm(x) + 1e-6)
        def Lf_h(x): return float(np.dot(-x / (np.linalg.norm(x) + 1e-6), -x))
        def Lg_h(x): return np.zeros(3)

        u_nominal = np.array([0.1, 0.1, 0.1])

        synthesizer = CBFSynthesizer(alpha=1.0)
        result = synthesizer.compute_safe_control(h, grad_h, Lf_h, Lg_h, u_nominal)

        assert result.is_safe or result.h_value >= 0
        assert len(result.u) == len(u_nominal)
        assert result.computation_time_ms >= 0

    def test_composed_cbf(self):
        """Test composed CBF from multiple constraints."""
        constraints = [
            lambda x: 1.0 - abs(x[0]),
            lambda x: 1.0 - abs(x[1]),
            lambda x: 1.0 - abs(x[2])
        ]
        grad_constraints = [
            lambda x: np.array([-1.0 if x[0] < 0 else 1.0, 0.0, 0.0]),
            lambda x: np.array([0.0, -1.0 if x[1] < 0 else 1.0, 0.0]),
            lambda x: np.array([0.0, 0.0, -1.0 if x[2] < 0 else 1.0])
        ]

        synthesizer = CBFSynthesizer()
        h_comp, grad_h_comp = synthesizer.synthesize_composed_cbf(
            constraints, grad_constraints
        )

        origin = np.zeros(3)
        assert h_comp(origin) == 1.0, "All constraints satisfied at origin"

    def test_zeroing_cbf(self):
        """Test zeroing CBF synthesis."""
        def constraint(x): return 1.0 - np.linalg.norm(x)
        def grad_constraint(x): return -x / (np.linalg.norm(x) + 1e-6)
        def f(x, u): return -x
        def g(x, u): return np.eye(3)

        synthesizer = CBFSynthesizer()
        h_zcbf, grad_h_zcbf = synthesizer.synthesize_zeroing_cbf(
            constraint, grad_constraint, f, g
        )

        assert h_zcbf is not None
        assert grad_h_zcbf is not None

    def test_lie_derivatives(self):
        """Test Lie derivative computation."""
        def h(x): return 1.0 - x[0]**2
        def grad_h(x): return np.array([-2 * x[0], 0.0, 0.0])
        def f(x, u): return np.array([-x[0], -x[1], -x[2]])

        synthesizer = CBFSynthesizer()
        x = np.array([0.5, 0.0, 0.0])

        Lf_h = synthesizer.compute_lie_derivative(h, grad_h, f, x)
        expected = float(np.dot(grad_h(x), f(x, None)))

        assert abs(Lf_h - expected) < 1e-10


# ============================================================================
# RUNTIME MONITOR TESTS
# ============================================================================

class TestRuntimeMonitor:
    """Tests for real-time safety monitoring."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)

        assert monitor.safety_bounds is not None
        assert monitor.buffer_head == 0
        assert monitor.buffer_count == 0

    def test_safe_state_update(self):
        """Test update with safe state."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])

        result = monitor.update(state)

        assert result.is_safe
        assert result.violation_risk >= 0
        assert result.monitor_time_us >= 0

    def test_boundary_violation(self):
        """Test detection of boundary violation."""
        bounds = SafetyBounds(
            x_min=-1 * np.ones(3),
            x_max=1 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = monitor.update(state)

        assert not result.is_safe

    def test_velocity_violation(self):
        """Test detection of velocity limit violation."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=1 * np.ones(3),
            a_max=2 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = monitor.update(state)

        assert not result.is_safe

    def test_custom_constraint(self):
        """Test custom safety constraint."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)

        def custom_constraint(state):
            return 1.0 - np.linalg.norm(state[:3])

        monitor.add_constraint(custom_constraint, "distance_from_origin")
        state = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = monitor.update(state)

        constraints = monitor.check_constraints(state)
        assert len(constraints) >= 3 # Built-in + custom

    def test_violation_prediction(self):
        """Test future violation prediction."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=10 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.array([4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        will_violate, time_to, step = monitor.predict_violation(state, dt=0.5, horizon=10)

        assert isinstance(will_violate, bool)
        assert time_to >= 0 or step == -1

    def test_latency_tracking(self):
        """Test latency tracking."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.zeros(9)

        for _ in range(100):
            monitor.update(state)

        stats = monitor.get_statistics()

        assert stats["total_updates"] == 100
        assert stats["mean_latency_us"] >= 0
        assert stats["max_latency_us"] >= 0

    def test_sub_millisecond_latency(self):
        """Test that latency is typically sub-millisecond."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.zeros(9)

        # Run many iterations to get stable measurement
        latencies = []
        for _ in range(1000):
            result = monitor.update(state)
            latencies.append(result.monitor_time_us)

        mean_latency = np.mean(latencies)
        assert mean_latency < 1000, f"Mean latency {mean_latency}us should be < 1ms"

    def test_state_history(self):
        """Test state history retrieval."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )

        monitor = RuntimeMonitor(bounds)
        state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(20):
            monitor.update(state)

        history = monitor.get_state_history(10)

        assert history.shape[0] <= 10
        assert history.shape[1] == 3


# ============================================================================
# SAFE PLANNER TESTS
# ============================================================================

class TestSafePlanner:
    """Tests for safe trajectory planning."""

    def test_planner_initialization(self):
        """Test planner initialization."""
        config = PlannerConfig(planner_type=PlannerType.MPCC)
        planner = SafePlanner(config)

        assert planner.config is not None
        assert len(planner.safety_constraints) == 0

    def test_hermitian_planning(self):
        """Test Hermite interpolation planning."""
        config = PlannerConfig(planner_type=PlannerType.HERMITIAN)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 1.0, 1.0])

        trajectory = planner.plan(start, goal)

        assert trajectory.positions is not None
        assert len(trajectory.positions) > 0
        assert np.allclose(trajectory.positions[0], start)
        assert np.allclose(trajectory.positions[-1], goal)
        assert trajectory.cost >= 0

    def test_mpcc_planning(self):
        """Test MPCC planning."""
        config = PlannerConfig(planner_type=PlannerType.MPCC, horizon=50)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.0, 0.0])

        trajectory = planner.plan(start, goal)

        assert trajectory.positions is not None
        assert len(trajectory.positions) == 51 # horizon + 1

    def test_rrt_planning(self):
        """Test RRT planning."""
        config = PlannerConfig(planner_type=PlannerType.RRT, max_iterations=100)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 1.0, 1.0])

        trajectory = planner.plan(start, goal)

        assert trajectory.positions is not None

    def test_ilqr_planning(self):
        """Test iLQR planning."""
        config = PlannerConfig(planner_type=PlannerType.iLQR, horizon=30)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.0, 0.0])

        trajectory = planner.plan(start, goal)

        assert trajectory.positions is not None

    def test_safety_constraint_integration(self):
        """Test planning with safety constraints."""
        config = PlannerConfig(planner_type=PlannerType.MPCC)
        planner = SafePlanner(config)

        # Add sphere constraint
        def safety_constraint(x):
            return 1.0 - np.linalg.norm(x - np.array([0.5, 0.5, 0.5]))

        planner.add_safety_constraint(safety_constraint)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 1.0, 1.0])

        trajectory = planner.plan(start, goal)

        assert trajectory.safety_score >= 0
        assert trajectory.safety_score <= 1.0

    def test_trajectory_resampling(self):
        """Test trajectory resampling."""
        config = PlannerConfig(planner_type=PlannerType.HERMITIAN, dt=0.01)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.0, 0.0])

        trajectory = planner.plan(start, goal)
        resampled = planner.resample_trajectory(trajectory, 0.02)

        assert resampled.positions is not None
        assert resampled.times[-1] == pytest.approx(trajectory.times[-1])

    def test_velocity_constraints(self):
        """Test that planned trajectories respect velocity limits."""
        config = PlannerConfig(
            planner_type=PlannerType.MPCC,
            max_velocity=1.0,
            horizon=50
        )
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.0, 0.0])

        trajectory = planner.plan(start, goal)

        velocities = trajectory.velocities
        max_vel = np.max(np.linalg.norm(velocities, axis=1))

        assert max_vel <= config.max_velocity * 1.1 # Small tolerance

    def test_initial_velocity(self):
        """Test planning with initial velocity."""
        config = PlannerConfig(planner_type=PlannerType.HERMITIAN)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.0, 0.0])

        trajectory = planner.plan(start, goal, initial_velocity)

        assert np.allclose(trajectory.velocities[0], initial_velocity, atol=0.5)


# ============================================================================
# COLLISION DETECTOR TESTS
# ============================================================================

class TestCollisionDetector:
    """Tests for collision detection."""

    def test_detector_initialization(self):
        """Test collision detector initialization."""
        detector = CollisionDetector(spatial_resolution=0.1)

        assert detector.spatial_resolution == 0.1
        assert len(detector.obstacles) == 0

    def test_sphere_obstacle(self):
        """Test sphere obstacle addition."""
        detector = CollisionDetector()
        center = np.array([1.0, 0.0, 0.0])
        radius = 0.5

        obs_id = detector.add_sphere_obstacle(center, radius)

        assert obs_id == 0
        assert len(detector.obstacles) == 1
        assert detector.obstacles[0]["type"] == "sphere"

    def test_box_obstacle(self):
        """Test box obstacle addition."""
        detector = CollisionDetector()
        center = np.array([1.0, 0.0, 0.0])
        half_extents = np.array([0.5, 0.5, 0.5])

        obs_id = detector.add_box_obstacle(center, half_extents)

        assert obs_id == 0
        assert detector.obstacles[0]["type"] == "box"

    def test_cylinder_obstacle(self):
        """Test cylinder obstacle addition."""
        detector = CollisionDetector()
        center = np.array([0.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])

        obs_id = detector.add_cylinder_obstacle(center, axis,0.5, 1.0)

        assert obs_id == 0
        assert detector.obstacles[0]["type"] == "cylinder"

    def test_point_cloud_obstacle(self):
        """Test point cloud obstacle addition."""
        detector = CollisionDetector()
        points = np.random.randn(100, 3) * 0.1

        obs_id = detector.add_point_cloud(points, radius=0.05)

        assert obs_id == 0
        assert detector.obstacles[0]["type"] == "point_cloud"

    def test_collision_detection_no_collision(self):
        """Test collision detection with no collision."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([2.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([0.0, 0.0, 0.0]), radius=0.1)

        assert not result.is_collision
        assert result.distance > 0
        assert result.check_time_us >= 0

    def test_collision_detection_with_collision(self):
        """Test collision detection with collision."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([0.3, 0.0, 0.0]), radius=0.3)

        assert result.is_collision
        assert result.distance < 0

    def test_collision_with_box(self):
        """Test collision with box obstacle."""
        detector = CollisionDetector()
        detector.add_box_obstacle(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))

        # Inside box
        result_inside = detector.check_collision(np.array([0.0, 0.0, 0.0]), radius=0.1)
        assert result_inside.is_collision

        # Outside box
        result_outside = detector.check_collision(np.array([2.0, 0.0, 0.0]), radius=0.1)
        assert not result_outside.is_collision

    def test_collision_with_cylinder(self):
        """Test collision with cylinder obstacle."""
        detector = CollisionDetector()
        detector.add_cylinder_obstacle(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.5, 2.0)

        # Inside cylinder
        result_inside = detector.check_collision(np.array([0.2, 0.0, 0.0]), radius=0.1)
        assert result_inside.is_collision

        # Outside cylinder
        result_outside = detector.check_collision(np.array([2.0, 0.0, 0.0]), radius=0.1)
        assert not result_outside.is_collision

    def test_point_cloud_collision(self):
        """Test point cloud collision detection."""
        detector = CollisionDetector()

        # Create point cloud in a sphere
        n_points = 100
        angles = np.random.uniform(0, 2*np.pi, n_points)
        heights = np.random.uniform(-0.5, 0.5, n_points)
        radii = np.random.uniform(0.8, 1.0, n_points)
        points = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
            heights
        ])

        detector.add_point_cloud(points, radius=0.05)

        # Point inside cloud
        result_inside = detector.check_collision(np.array([0.9, 0.0, 0.0]), radius=0.05)
        # May or may not be in collision depending on random points

        # Point far away
        result_outside = detector.check_collision(np.array([5.0, 0.0, 0.0]), radius=0.05)
        assert not result_outside.is_collision

    def test_trajectory_collision_check(self):
        """Test trajectory collision checking."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        # Trajectory that passes through obstacle
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ])

        has_collision, first_idx, collision_indices, time_us = detector.check_trajectory_collision(
            positions, radius=0.1
        )

        assert has_collision
        # The sphere at x=1 has radius 0.5 and the agent radius is 0.1,
        # so the trajectory first intersects it at x=0.5 (index 1).
        assert first_idx == 1
        assert 1 in collision_indices
        assert time_us >= 0

    def test_ignore_obstacle_ids(self):
        """Test ignoring specific obstacles."""
        detector = CollisionDetector()
        obs_id = detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([0.0, 0.0, 0.0]), radius=0.1, ignore_ids=[obs_id])

        assert not result.is_collision

    def test_sdf_computation(self):
        """Test signed distance field computation."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 1.0)

        sdf = detector.build_sdf(resolution=0.1)

        # Inside sphere
        inside_dist = sdf(np.array([0.0, 0.0, 0.0]))
        assert inside_dist < 0

        # On surface
        surface_dist = sdf(np.array([1.0, 0.0, 0.0]))
        assert abs(surface_dist) < 0.1

        # Outside sphere
        outside_dist = sdf(np.array([2.0, 0.0, 0.0]))
        assert outside_dist > 0

    def test_closest_point(self):
        """Test closest point computation."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([2.0, 0.0, 0.0]), radius=0.1)

        assert result.closest_point is not None
        assert np.allclose(result.closest_point, np.array([1.5, 0.0, 0.0]))

    def test_collision_normal(self):
        """Test collision normal computation."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([2.0, 0.0, 0.0]), radius=0.1)

        assert result.collision_normal is not None
        assert abs(np.linalg.norm(result.collision_normal) - 1.0) < 1e-6

    def test_clear_obstacles(self):
        """Test clearing obstacles."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 0.5)
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        assert len(detector.obstacles) == 2

        detector.clear_obstacles()

        assert len(detector.obstacles) == 0
        assert len(detector.grid) == 0

    def test_statistics(self):
        """Test statistics retrieval."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 0.5)
        detector.add_box_obstacle(np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))

        stats = detector.get_statistics()

        assert stats["num_obstacles"] == 2
        assert "sphere" in stats["obstacle_types"]
        assert "box" in stats["obstacle_types"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for combined components."""

    def test_full_safety_pipeline(self):
        """Test full safety verification pipeline."""
        # 1. Create Lyapunov verifier
        verifier = LyapunovVerifier(grid_resolution=30)
        P = np.eye(3)

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x

        lyap_result = verifier.verify_lyapunov_function(V, grad_V, f, domain_radius=5.0)

        # 2. Create CBF synthesizer
        synthesizer = CBFSynthesizer()
        h, grad_h = synthesizer.synthesize_exponential_cbf(
            lambda x: 1.0 - np.linalg.norm(x),
            lambda x: -x / (np.linalg.norm(x) + 1e-6),
            decay_rate=1.0
        )

        # 3. Create runtime monitor
        bounds = SafetyBounds(
            x_min=-2 * np.ones(3),
            x_max=2 * np.ones(3),
            v_max=1 * np.ones(3),
            a_max=0.5 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)

        # 4. Create collision detector
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.5, 0.5, 0.5]), 0.3)

        # 5. Create planner
        planner = SafePlanner(PlannerConfig(planner_type=PlannerType.MPCC))
        trajectory = planner.plan(np.zeros(3), np.array([1.0, 1.0, 1.0]))

        # Verify all components work together
        assert lyap_result.is_lyapunov
        assert trajectory.positions is not None

        # Test monitoring
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        monitor_result = monitor.update(state)
        assert monitor_result.is_safe

        # Test collision detection
        collision_result = detector.check_collision(np.array([0.0, 0.0, 0.0]), radius=0.1)
        assert not collision_result.is_collision

    def test_cbf_with_runtime_monitor(self):
        """Test CBF control with runtime monitoring."""
        synthesizer = CBFSynthesizer(alpha=1.0)

        def h(x): return 1.0 - np.linalg.norm(x[:3])
        def grad_h(x): return -x[:3] / (np.linalg.norm(x[:3]) + 1e-6)
        def Lf_h(x): return float(np.dot(-x[:3] / (np.linalg.norm(x[:3]) + 1e-6), -x[:3]))
        def Lg_h(x): return np.zeros(3)

        bounds = SafetyBounds(
            x_min=-2 * np.ones(3),
            x_max=2 * np.ones(3),
            v_max=1 * np.ones(3),
            a_max=0.5 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)

        # Simulate closed-loop control
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u_nominal = np.array([0.1, 0.1, 0.1])

        for _ in range(10):
            control_result = synthesizer.compute_safe_control(h, grad_h, Lf_h, Lg_h, u_nominal)

            # Update state (simple integration)
            state[:3] += state[3:6] * 0.01
            state[3:6] += control_result.u * 0.01

            # Monitor
            monitor_result = monitor.update(state)

            # Check safety
            if not monitor_result.is_safe:
                break

    def test_collision_aware_planning(self):
        """Test planning with collision avoidance."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.5, 0.0, 0.0]), 0.3)

        def collision_avoidance(x):
            result = detector.check_collision(x, radius=0.1)
            return -result.distance if result.distance < 0.5 else 0.5

        config = PlannerConfig(planner_type=PlannerType.MPCC)
        planner = SafePlanner(config, safety_constraints=[collision_avoidance])

        trajectory = planner.plan(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        )

        assert trajectory is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    def test_lyapunov_verification_performance(self):
        """Test Lyapunov verification speed."""
        verifier = LyapunovVerifier(grid_resolution=100)
        P = np.eye(3)

        def V(x): return float(x.T @ P @ x)
        def grad_V(x): return 2 * P @ x
        def f(x, u): return -x

        start_time = time.perf_counter()
        result = verifier.verify_lyapunov_function(V, grad_V, f, domain_radius=10.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 5000, f"Lyapunov verification took {elapsed_ms}ms, expected< 5000ms"

    def test_cbf_synthesis_performance(self):
        """Test CBF synthesis speed."""
        synthesizer = CBFSynthesizer()

        def constraint(x): return 1.0 - np.linalg.norm(x)
        def grad_constraint(x): return -x / (np.linalg.norm(x) + 1e-6)
        def f(x, u): return -x
        def g(x, u): return np.eye(3)

        start_time = time.perf_counter()
        result = synthesizer.synthesize_from_spec(constraint, grad_constraint, f, g)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, f"CBF synthesis took {elapsed_ms}ms, expected < 100ms"

    def test_collision_check_performance(self):
        """Test collision checking speed."""
        detector = CollisionDetector()
        for i in range(10):
            detector.add_sphere_obstacle(
                np.array([i * 0.5, 0.0, 0.0]),
                0.2
            )

        n_checks = 10000
        start_time = time.perf_counter()

        for i in range(n_checks):
            x = np.array([np.random.randn() * 2, np.random.randn() * 2, np.random.randn() * 2])
            detector.check_collision(x, radius=0.05)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        avg_time_us = elapsed_ms * 1000 / n_checks

        assert avg_time_us < 100, f"Avg collision check took {avg_time_us}us, expected < 100us"

    def test_mpcc_planning_performance(self):
        """Test MPCC planning speed."""
        config = PlannerConfig(planner_type=PlannerType.MPCC, horizon=100)
        planner = SafePlanner(config)

        start_time = time.perf_counter()
        trajectory = planner.plan(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0])
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 1000, f"MPCC planning took {elapsed_ms}ms, expected < 1000ms"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and corner conditions."""

    def test_zero_state(self):
        """Test handling of zero state."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)
        result = monitor.update(np.zeros(9))

        assert result.is_safe

    def test_degenerate_lyapunov(self):
        """Test degenerate case in Lyapunov verification."""
        # Degenerate P matrix
        P = np.zeros((3, 3))

        def V(x): return 0.0
        def grad_V(x): return np.zeros(3)
        def f(x, u): return -x

        verifier = LyapunovVerifier()
        result = verifier._check_positive_definite(V, np.random.randn(100, 3))

        assert not result["is_positive_definite"]

    def test_nan_handling(self):
        """Test handling of NaN values."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)

        state = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = monitor.update(state)

        assert not result.is_safe

    def test_empty_point_cloud(self):
        """Test with empty point cloud."""
        detector = CollisionDetector()
        detector.add_point_cloud(np.zeros((0, 3)), radius=0.05)

        result = detector.check_collision(np.array([0.0, 0.0, 0.0]), radius=0.1)

        assert not result.is_collision

    def test_identical_start_goal(self):
        """Test planning with identical start and goal."""
        config = PlannerConfig(planner_type=PlannerType.HERMITIAN)
        planner = SafePlanner(config)

        point = np.array([0.0, 0.0, 0.0])
        trajectory = planner.plan(point, point)

        assert np.allclose(trajectory.positions[0], point)
        assert np.allclose(trajectory.positions[-1], point)


# ============================================================================
# REGRESSION TESTS (bug fixes)
# ============================================================================

class TestRegressions:
    """Regression tests for specific bugs that were fixed.

    Each test pins the corrected behaviour so the same defect cannot
    silently reappear.
    """

    @pytest.mark.parametrize("planner_type", list(PlannerType))
    def test_all_planners_reach_goal(self, planner_type):
        """Every planner must land on the goal within the horizon.

        MPCC used to scale desired velocity by distance-to-goal, which
        made it asymptotically slow and it stopped ~0.5 short.
        """
        config = PlannerConfig(planner_type=planner_type, horizon=100, dt=0.01)
        planner = SafePlanner(config)

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 1.0, 1.0])

        trajectory = planner.plan(start, goal)

        assert np.allclose(trajectory.positions[0], start)
        assert np.allclose(trajectory.positions[-1], goal, atol=1e-6)
        assert np.all(np.isfinite(trajectory.positions)), "trajectory must be finite"

    @pytest.mark.parametrize("planner_type", list(PlannerType))
    def test_no_runtime_overflow_warnings(self, planner_type):
        """iLQR used to shrink its regularizer to zero and overflow."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            config = PlannerConfig(planner_type=planner_type, horizon=50)
            planner = SafePlanner(config)
            trajectory = planner.plan(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0])
            )

        assert np.all(np.isfinite(trajectory.positions))

    @pytest.mark.parametrize("n_dim", [2, 3, 5, 7])
    def test_lyapunov_verifier_arbitrary_dimension(self, n_dim):
        """The verifier must work for any state dimension.

        The grid generator used to hard-code 3-D samples, so 2-D and
        5-D systems raised a matmul dimension error.

        1-D is excluded: a 1-D "ball" is an interval, and random samples
        can land arbitrarily close to the origin, which makes the
        ``min_V > tolerance`` positive-definiteness check flaky rather
        than wrong. It carries no extra signal for this regression.
        """
        P = np.eye(n_dim)

        def V(x):
            return float(x.T @ P @ x)

        def grad_V(x):
            return 2 * P @ x

        def f(x, u):
            return -x

        verifier = LyapunovVerifier(grid_resolution=10)
        result = verifier.verify_lyapunov_function(
            V, grad_V, f, domain_radius=5.0
        )

        assert result.is_stable
        assert result.details["positive_definite"]["is_positive_definite"]

    def test_runtime_monitor_rejects_nonfinite_state(self):
        """NaN in the state must never be reported as safe.

        Comparisons against NaN are all False, so the old fast path
        returned is_safe=True for a NaN observation.
        """
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)

        for bad_state in (
            np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([np.nan] * 9),
        ):
            result = monitor.update(bad_state)
            assert not result.is_safe
            assert result.violation_risk == 1.0

    def test_broad_phase_does_not_miss_far_obstacles(self):
        """The broad phase must keep obstacles it cannot prove are far.

        The spatial-hash broad phase only looked at a 3x3x3 cell block,
        so obstacles outside that window vanished entirely.
        """
        detector = CollisionDetector()
        # Obstacle whose only grid sample is 1.0 away from the query --
        # far outside a single 0.1-resolution cell.
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(np.array([2.0, 0.0, 0.0]), radius=0.1)

        assert result.closest_point is not None, "nearest obstacle must be reported"
        assert result.collision_normal is not None
        assert not result.is_collision

    def test_broad_phase_keeps_all_touching_obstacles(self):
        """Every obstacle that can touch the agent must be a candidate."""
        detector = CollisionDetector()
        ids = [
            detector.add_sphere_obstacle(np.array([float(i), 0.0, 0.0]), 0.2)
            for i in range(10)
        ]

        for i in range(10):
            result = detector.check_collision(
                np.array([float(i), 0.0, 0.0]), radius=0.05
            )
            assert result.is_collision, f"obstacle {ids[i]} should collide"

    def test_broad_phase_ignores_requested_ids(self):
        """ignore_ids must still be honoured after the broad-phase fix."""
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([0.0, 0.0, 0.0]), 0.5)

        result = detector.check_collision(
            np.array([0.0, 0.0, 0.0]), radius=0.1, ignore_ids=[0]
        )

        assert not result.is_collision

    def test_exponential_cbf_zero_level_set(self):
        """h_exp must be positive inside and zero on the boundary.

        The old assertion compared against radius**2, which mixed up the
        area-form constraint with a radius-form one.
        """
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0

        h, grad_h = CBFSynthesizer.synthesize_exponential_cbf(
            lambda x: radius**2 - np.sum((x - center)**2),
            lambda x: -2 * (x - center),
            decay_rate=1.0
        )

        assert h(np.array([0.0, 0.0, 0.0])) > 0
        assert abs(h(np.array([1.0, 0.0, 0.0]))) < 1e-10
        assert np.allclose(grad_h(np.array([1.0, 0.0, 0.0])), -2 * (np.array([1.0, 0.0, 0.0]) - center))

    def test_cbf_synthesis_static_callable_from_class(self):
        """The synthesize_* helpers must be callable without an instance.

        They never used ``self`` but were declared as instance methods,
        so ``CBFSynthesizer.synthesize_exponential_cbf(...)`` raised
        TypeError.
        """
        for method in (
            CBFSynthesizer.synthesize_exponential_cbf,
            CBFSynthesizer.synthesize_composed_cbf,
        ):
            assert callable(method)

        # Both must work on the class itself.
        # h_exp(x) = h(x) * exp(decay * h(x)), so at the origin with
        # h = 1 and decay = 1 the value is e, and the zero level set is
        # preserved (h = 0 => h_exp = 0).
        h, grad_h = CBFSynthesizer.synthesize_exponential_cbf(
            lambda x: 1.0 - np.sum(x**2),
            lambda x: -2 * x,
            decay_rate=1.0
        )
        assert h(np.zeros(3)) == pytest.approx(np.e)
        assert abs(h(np.array([1.0, 0.0, 0.0]))) < 1e-10

        h2, grad_h2 = CBFSynthesizer.synthesize_composed_cbf(
            [lambda x: 1.0 - np.sum(x**2)],
            [lambda x: -2 * x]
        )
        assert h2(np.zeros(3)) == pytest.approx(1.0)

    def test_trajectory_first_collision_index(self):
        """First collision index must reflect the real geometry.

        Sphere at x=1 with radius 0.5 and agent radius 0.1 means the
        trajectory first enters it at x=0.5 (index 1), not index 2.
        """
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ])

        has_collision, first_idx, collision_indices, time_us = \
            detector.check_trajectory_collision(positions, radius=0.1)

        assert has_collision
        assert first_idx == 1
        assert set(collision_indices) == {1, 2, 3}

    # -- cbf_synthesizer edge cases --------------------------------------

    def test_compute_safe_control_accepts_scalar_lie_derivative(self):
        """A plain-float Lg_h must not crash compute_safe_control.

        The old code did ``len(Lg_val.shape)`` on a Python float and
        raised AttributeError whenever Lg_h returned a scalar.
        """
        synthesizer = CBFSynthesizer(alpha=1.0)

        result = synthesizer.compute_safe_control(
            h=lambda x: 1.0,
            grad_h=lambda x: np.zeros(3),
            Lf_h=lambda x: 0.0,
            Lg_h=lambda x: 0.0,          # scalar, not an array
            u_nominal=np.zeros(3)
        )

        assert result.u.shape == (3,)
        assert np.all(np.isfinite(result.u))
        assert 0.0 <= result.constraint_satisfaction <= 1.0

    def test_compute_safe_control_accepts_various_lg_shapes(self):
        """Lg_h as scalar, 0-d array, 1-D and (1, n) row must all work."""
        synthesizer = CBFSynthesizer(alpha=1.0)
        u_nominal = np.array([0.0, 0.0, 0.0])

        for lg in (0.5, np.float64(0.5), np.array(0.5),
                   np.array([1.0, 1.0, 1.0]), np.array([[1.0, 1.0, 1.0]])):
            result = synthesizer.compute_safe_control(
                h=lambda x: 1.0,
                grad_h=lambda x: np.zeros(3),
                Lf_h=lambda x: 0.0,
                Lg_h=lambda x, _lg=lg: _lg,
                u_nominal=u_nominal
            )
            assert result.u.shape == (3,), f"failed for Lg_h={lg!r}"

    def test_composed_cbf_rejects_empty_and_mismatched_inputs(self):
        """Empty / mismatched inputs must raise a clear ValueError.

        ``min([])`` used to surface as an opaque ValueError from deep
        inside the closure.
        """
        with pytest.raises(ValueError, match="at least one constraint"):
            CBFSynthesizer.synthesize_composed_cbf([], [])

        with pytest.raises(ValueError, match="matching order"):
            CBFSynthesizer.synthesize_composed_cbf(
                [lambda x: 1.0], [lambda x: np.zeros(3), lambda x: np.zeros(3)]
            )

        with pytest.raises(ValueError, match="positive finite"):
            CBFSynthesizer.synthesize_composed_cbf(
                [lambda x: 1.0, lambda x: 1.0],
                [lambda x: np.zeros(3), lambda x: np.zeros(3)],
                weights=np.array([0.0, 0.0])
            )

    def test_exponential_cbf_survives_negative_decay_rate(self):
        """A negative decay_rate must not overflow the exponential.

        ``h * exp(decay * h)`` with decay < 0 and h << 0 used to blow up
        to ~1e87, destroying the sign the CBF relies on.
        """
        h, grad_h = CBFSynthesizer.synthesize_exponential_cbf(
            lambda x: 1.0 - np.sum(x**2),
            lambda x: -2 * x,
            decay_rate=-2.0
        )

        far = np.array([10.0, 0.0, 0.0])   # h = -99
        val = h(far)

        assert np.isfinite(val), "h_exp must stay finite"
        assert val < 0, "h_exp must keep the sign of h"
        assert np.all(np.isfinite(grad_h(far)))

        # Inside the safe set the sign must still be positive.
        assert h(np.zeros(3)) > 0

    # -- runtime_monitor edge cases --------------------------------------

    def test_monitor_accepts_short_state(self):
        """A state shorter than the configured dimension must not crash.

        ``x < x_min`` on a truncated slice used to raise a broadcast
        ValueError.
        """
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)
        short = np.array([1.0, 0.5])   # only 2 of 3 dims

        assert np.isfinite(monitor.update(short).violation_risk)
        assert len(monitor.check_constraints(short)) >= 2
        assert monitor.predict_violation(short)[0] in (True, False)

    def test_monitor_rejects_violating_short_state(self):
        """Padding must not hide a genuine violation in the given part."""
        bounds = SafetyBounds(
            x_min=-5 * np.ones(3),
            x_max=5 * np.ones(3),
            v_max=2 * np.ones(3),
            a_max=1 * np.ones(3)
        )
        monitor = RuntimeMonitor(bounds)

        # First entry is far outside x_max=5 -> must be unsafe.
        assert not monitor.update(np.array([99.0])).is_safe

    # -- safe_planner edge cases -----------------------------------------

    def test_planner_non_positive_horizon(self):
        """horizon <= 0 must not divide by zero.

        ``v0 = (goal - start) / (horizon * dt)`` warned and produced
        inf/nan for horizon = 0.
        """
        for horizon in (0, -5):
            for planner_type in PlannerType:
                config = PlannerConfig(planner_type=planner_type, horizon=horizon)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    trajectory = SafePlanner(config).plan(
                        np.zeros(3), np.ones(3)
                    )
                assert len(trajectory.positions) >= 2
                assert np.all(np.isfinite(trajectory.positions))

    def test_collision_detector_negative_coordinates(self):
        """Negative coordinates must collide correctly.

        Guards the spatial-hash key path (``int(x / res)`` truncates
        toward zero for negatives).
        """
        detector = CollisionDetector()
        detector.add_sphere_obstacle(np.array([-1.0, -1.0, -1.0]), 0.5)

        assert detector.check_collision(
            np.array([-1.2, -1.0, -1.0]), radius=0.1
        ).is_collision

        # Distance sqrt(3) - 0.5 = 1.23 >> 0.6, so this must NOT collide.
        assert not detector.check_collision(
            np.array([0.0, 0.0, 0.0]), radius=0.1
        ).is_collision

        detector2 = CollisionDetector()
        detector2.add_sphere_obstacle(np.zeros(3), 0.5)
        assert detector2.check_collision(
            np.array([-0.3, -0.2, -0.1]), radius=0.1
        ).is_collision


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])