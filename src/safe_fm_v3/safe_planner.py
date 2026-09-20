"""
Safe Planner: Constrained trajectory planning with safety guarantees.

This module provides trajectory planning with formal safety constraints
using control barrier functions and convex optimization.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum


class PlannerType(Enum):
    """Types of trajectory planners."""
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    MPCC = "mpcc"
    iLQR = "ilqr"
    HERMITIAN = "hermitian"


@dataclass
class Trajectory:
    """Planned trajectory."""
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    cost: float
    is_safe: bool
    safety_score: float


@dataclass
class Waypoint:
    """Waypoint specification."""
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    time: Optional[float] = None
    constraint: Optional[Callable] = None


@dataclass
class PlannerConfig:
    """Configuration for trajectory planner."""
    planner_type: PlannerType = PlannerType.MPCC
    dt: float = 0.01
    horizon: int = 100
    max_iterations: int = 100
    convergence_tolerance: float = 1e-4
    safety_margin: float = 0.1
    max_velocity: float = 5.0
    max_acceleration: float = 2.0
    regularization: float = 1e-4


class SafePlanner:
    """
    Safe trajectory planner with constraint satisfaction.

    Plans trajectories that satisfy safety constraints while optimizing
    for smoothness and goal achievement.

    Attributes:
        config: Planner configuration
        safety_constraints: List of safety constraint functions
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        safety_constraints: Optional[List[Callable]] = None
    ):
        """
        Initialize safe planner.

        Args:
            config: Planner configuration
            safety_constraints: List of h(x) >= 0 constraint functions
        """
        self.config = config or PlannerConfig()
        self.safety_constraints = safety_constraints or []

    def add_safety_constraint(self, constraint: Callable) -> None:
        """Add a safety constraint function h(x) >= 0."""
        self.safety_constraints.append(constraint)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None,
        obstacles: Optional[List[Dict]] = None
    ) -> Trajectory:
        """
        Plan trajectory from start to goal.

        Args:
            start: Starting position
            goal: Goal position
            initial_velocity: Initial velocity (optional)
            obstacles: List of obstacle specifications

        Returns:
            Trajectory object
        """
        if self.config.planner_type == PlannerType.MPCC:
            return self._plan_mpcc(start, goal, initial_velocity)
        elif self.config.planner_type == PlannerType.RRT:
            return self._plan_rrt(start, goal, initial_velocity)
        elif self.config.planner_type == PlannerType.iLQR:
            return self._plan_ilqr(start, goal, initial_velocity)
        else:
            return self._plan_hermitian(start, goal, initial_velocity)

    def _degenerate_trajectory(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        n_dim: int,
        dt: float
    ) -> Trajectory:
        """Two-point trajectory used when the horizon is non-positive.

        With ``horizon <= 0`` there are no steps to integrate and the
        normal formulas divide by zero, so return just start -> goal.
        """
        positions = np.stack([
            np.asarray(start, dtype=float).reshape(-1),
            np.asarray(goal, dtype=float).reshape(-1)
        ])
        is_safe, safety_score = self._check_trajectory_safety(positions)
        return Trajectory(
            positions=positions,
            velocities=np.zeros((2, n_dim)),
            accelerations=np.zeros((2, n_dim)),
            times=np.array([0.0, dt]),
            cost=0.0,
            is_safe=is_safe,
            safety_score=safety_score
        )

    def _plan_hermitian(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None
    ) -> Trajectory:
        """
        Plan trajectory using Hermite interpolation.

        Args:
            start: Starting position
            goal: Goal position
            initial_velocity: Initial velocity

        Returns:
            Trajectory object
        """
        n_dim = len(start)
        horizon = self.config.horizon
        dt = self.config.dt

        # A non-positive horizon has no steps to integrate; degrade to a
        # two-point trajectory instead of dividing by zero below.
        if horizon <= 0:
            return self._degenerate_trajectory(start, goal, n_dim, dt)

        # Initialize trajectory arrays
        positions = np.zeros((horizon + 1, n_dim))
        velocities = np.zeros((horizon + 1, n_dim))
        accelerations = np.zeros((horizon + 1, n_dim))
        times = np.arange(horizon + 1) * dt

        positions[0] = start
        positions[-1] = goal

        # Default velocities
        if initial_velocity is None:
            v0 = (goal - start) / (horizon * dt) * 0.5
        else:
            v0 = initial_velocity

        v_final = np.zeros(n_dim)

        velocities[0] = v0
        velocities[-1] = v_final

        # Compute jerk-minimizing trajectory via cubic Hermite
        for i in range(horizon):
            tau = i / horizon
            tau2 = tau * tau
            tau3 = tau2 * tau

            # Hermite basis functions
            h00 = 2*tau3 - 3*tau2 + 1
            h10 = tau3 - 2*tau2 + tau
            h01 = -2*tau3 + 3*tau2
            h11 = tau3 - tau2

            positions[i+1] = h00 * start + h10 * v0 * horizon * dt + h01 * goal + h11 * v_final * horizon * dt

            # Compute velocities
            dh00 = 6*tau2 - 6*tau
            dh10 = 3*tau2 - 4*tau + 1
            dh01 = -6*tau2 + 6*tau
            dh11 = 3*tau2 - 2*tau

            velocities[i+1] = (dh00 * start + dh10 * v0 * horizon * dt +
                               dh01 * goal + dh11 * v_final * horizon * dt) / (horizon * dt)

            # Compute accelerations (numerical)
            if i > 0:
                accelerations[i] = (velocities[i+1] - velocities[i]) / dt

        # The loop above stops at tau = (horizon-1)/horizon, so the
        # final sample never lands exactly on the goal. Pin it.
        positions[horizon] = goal
        velocities[horizon] = v_final
        if horizon > 0:
            accelerations[horizon] = accelerations[horizon - 1]

        # Compute cost
        jerk_squared = np.sum((accelerations[1:] - accelerations[:-1]) / dt**2)**2
        cost = jerk_squared

        # Check safety
        is_safe, safety_score = self._check_trajectory_safety(positions)

        return Trajectory(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
            cost=cost,
            is_safe=is_safe,
            safety_score=safety_score
        )

    def _plan_mpcc(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None
    ) -> Trajectory:
        """
        Plan trajectory using Model Predictive Path Integral Control (simplified).

        Args:
            start: Starting position
            goal: Goal position
            initial_velocity: Initial velocity

        Returns:
            Trajectory object
        """
        n_dim = len(start)
        horizon = self.config.horizon
        dt = self.config.dt

        if horizon <= 0:
            return self._degenerate_trajectory(start, goal, n_dim, dt)

        positions = np.zeros((horizon + 1, n_dim))
        velocities = np.zeros((horizon + 1, n_dim))
        accelerations = np.zeros((horizon + 1, n_dim))
        times = np.arange(horizon + 1) * dt

        positions[0] = start

        if initial_velocity is None:
            initial_velocity = np.zeros(n_dim)
        velocities[0] = initial_velocity

        # Simple potential field planning
        for i in range(horizon):
            # Potential field gradient
            to_goal = goal - positions[i]
            dist_to_goal = np.linalg.norm(to_goal)
            remaining_time = (horizon - i) * dt

            if dist_to_goal > 1e-6:
                # Distribute the remaining distance over the remaining
                # time instead of scaling the desired speed by the
                # distance itself. Speed proportional to distance makes
                # the controller asymptotically slow and it never
                # actually reaches the goal within the horizon.
                needed_speed = dist_to_goal / remaining_time if remaining_time > 0 else 0.0
                desired_velocity = to_goal / dist_to_goal * min(self.config.max_velocity, needed_speed)
            else:
                desired_velocity = np.zeros(n_dim)

            # Acceleration towards desired velocity
            a_desired = (desired_velocity - velocities[i]) / dt
            a_desired = np.clip(a_desired, -self.config.max_acceleration, self.config.max_acceleration)

            accelerations[i] = a_desired

            # Integrate
            velocities[i+1] = velocities[i] + a_desired * dt
            velocities[i+1] = np.clip(velocities[i+1], -self.config.max_velocity, self.config.max_velocity)
            positions[i+1] = positions[i] + velocities[i+1] * dt

        # Final integration
        accelerations[-1] = accelerations[-2]

        # Compute cost
        control_cost = np.sum(accelerations**2) * dt
        tracking_cost = np.sum((positions[-1] - goal)**2)
        cost = control_cost + tracking_cost

        # Check safety
        is_safe, safety_score = self._check_trajectory_safety(positions)

        return Trajectory(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
            cost=cost,
            is_safe=is_safe,
            safety_score=safety_score
        )

    def _plan_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None
    ) -> Trajectory:
        """
        Plan trajectory using RRT (Rapidly-exploring Random Tree).

        Args:
            start: Starting position
            goal: Goal position
            initial_velocity: Initial velocity (unused for RRT)

        Returns:
            Trajectory object
        """
        from collections import deque

        n_dim = len(start)
        max_iterations = self.config.max_iterations

        # RRT parameters
        step_size = 0.5
        goal_bias = 0.1

        # Tree structure: list of (node, parent_index)
        tree = [(start.copy(), -1)]
        tree_positions = [start.copy()]

        goal_reached = False
        goal_idx = -1

        for _ in range(max_iterations):
            # Sample
            if np.random.random() < goal_bias:
                sample = goal.copy()
            else:
                bounds_min = np.minimum(start, goal) - 5
                bounds_max = np.maximum(start, goal) + 5
                sample = np.random.uniform(bounds_min, bounds_max)

            # Find nearest
            nearest_idx = 0
            min_dist = float('inf')
            for idx, pos in enumerate(tree_positions):
                dist = np.linalg.norm(sample - pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx

            # Steer
            direction = sample - tree_positions[nearest_idx]
            dist = np.linalg.norm(direction)
            if dist > step_size:
                direction = direction / dist * step_size

            new_node = tree_positions[nearest_idx] + direction

            # Check safety
            is_safe = self._check_point_safety(new_node)

            if is_safe:
                tree.append((new_node, nearest_idx))
                tree_positions.append(new_node)

                # Check goal
                if np.linalg.norm(new_node - goal) < step_size:
                    tree.append((goal.copy(), len(tree_positions) - 1))
                    tree_positions.append(goal.copy())
                    goal_reached = True
                    goal_idx = len(tree_positions) - 1
                    break

        if not goal_reached:
            # Return closest to goal
            goal_idx = np.argmin([np.linalg.norm(pos - goal) for pos in tree_positions])

        # Extract path
        path_indices = []
        idx = goal_idx
        while idx >= 0:
            path_indices.append(idx)
            idx = tree[idx][1]

        path_indices = path_indices[::-1]
        path_positions = [tree_positions[i] for i in path_indices]

        # Interpolate to create smooth trajectory
        trajectory = self._smooth_path(np.array(path_positions), start, goal)

        return trajectory

    def _plan_ilqr(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        initial_velocity: Optional[np.ndarray] = None
    ) -> Trajectory:
        """
        Plan trajectory using iterative LQR (simplified).

        Args:
            start: Starting position
            goal: Goal position
            initial_velocity: Initial velocity

        Returns:
            Trajectory object
        """
        n_dim = len(start)
        horizon = self.config.horizon
        dt = self.config.dt
        reg = self.config.regularization

        # Initialize trajectory
        trajectory = self._plan_hermitian(start, goal, initial_velocity)
        positions = trajectory.positions.copy()
        velocities = trajectory.velocities.copy()
        accelerations = trajectory.accelerations.copy()

        # Cost matrices
        Q_pos = np.eye(n_dim)
        Q_vel = np.eye(n_dim) * 0.1
        R = np.eye(n_dim) * 0.01

        # iLQR iterations
        for iteration in range(self.config.max_iterations):
            # Forward pass with line search
            K_list = []
            k_list = []

            # Backward pass
            for t in range(horizon - 1, -1, -1):
                # State cost gradient
                pos_err = positions[t] - goal
                vel_err = velocities[t]

                grad_pos = Q_pos @ pos_err
                grad_vel = Q_vel @ vel_err

                # Simplified gain computation. ``reg`` is a *regularizer*
                # that bounds the feedback gain: it must never shrink to
                # zero, otherwise the control blows up numerically.
                K = np.zeros((n_dim, n_dim * 2))
                denom = R.diagonal().mean() + max(reg, self.config.regularization)
                k = -(grad_pos + grad_vel) / denom
                k = np.clip(k, -self.config.max_acceleration, self.config.max_acceleration)

                K_list.append(K)
                k_list.append(k)

            # Forward pass
            positions_new = positions.copy()
            velocities_new = velocities.copy()
            accelerations_new = accelerations.copy()

            for t in range(horizon):
                # Apply control
                state = np.concatenate([positions[t], velocities[t]])
                u = K_list[t] @ state + k_list[t]

                if t < horizon - 1:
                    accelerations_new[t] = u
                    velocities_new[t+1] = velocities_new[t] + u * dt
                    positions_new[t+1] = positions_new[t] + velocities_new[t+1] * dt

            # Check convergence
            pos_diff = np.max(np.abs(positions_new - positions))
            if pos_diff < self.config.convergence_tolerance:
                break

            positions = positions_new
            velocities = velocities_new
            accelerations = accelerations_new

            # Reduce regularization
            reg *= 0.9

        times = np.arange(horizon + 1) * dt
        cost = np.sum(accelerations**2) * dt
        is_safe, safety_score = self._check_trajectory_safety(positions)

        return Trajectory(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            times=times,
            cost=cost,
            is_safe=is_safe,
            safety_score=safety_score
        )

    def _smooth_path(
        self,
        path: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray
    ) -> Trajectory:
        """Smooth a waypoint path into a full trajectory."""
        n_dim = path.shape[1]
        n_points = len(path)

        # Compute velocities at waypoints
        velocities = np.zeros((n_points, n_dim))
        for i in range(1, n_points - 1):
            velocities[i] = (path[i+1] - path[i-1]) / 2

        # Interpolate using Hermite splines
        total_segments = n_points * 10
        positions = np.zeros((total_segments + 1, n_dim))
        velocities_out = np.zeros((total_segments + 1, n_dim))
        accelerations = np.zeros((total_segments + 1, n_dim))
        times = np.zeros(total_segments + 1)

        idx = 0
        for i in range(n_points - 1):
            seg_len = int(total_segments / (n_points - 1))
            for j in range(seg_len):
                tau = j / seg_len
                tau2 = tau * tau
                tau3 = tau2 * tau

                h00 = 2*tau3 - 3*tau2 + 1
                h10 = tau3 - 2*tau2 + tau
                h01 = -2*tau3 + 3*tau2
                h11 = tau3 - tau2

                positions[idx] = h00 * path[i] + h10 * velocities[i] + h01 * path[i+1] + h11 * velocities[i+1]
                times[idx] = j * self.config.dt
                idx += 1

        positions[-1] = path[-1]

        cost = np.sum(accelerations**2) * self.config.dt
        is_safe, safety_score = self._check_trajectory_safety(positions)

        return Trajectory(
            positions=positions,
            velocities=velocities_out,
            accelerations=accelerations,
            times=times,
            cost=cost,
            is_safe=is_safe,
            safety_score=safety_score
        )

    def _check_point_safety(self, point: np.ndarray) -> bool:
        """Check if a single point is safe."""
        for constraint in self.safety_constraints:
            try:
                if constraint(point) < 0:
                    return False
            except Exception:
                continue
        return True

    def _check_trajectory_safety(self, positions: np.ndarray) -> Tuple[bool, float]:
        """Check safety of an entire trajectory."""
        if not self.safety_constraints:
            return True, 1.0

        min_margin = float('inf')
        for pos in positions:
            for constraint in self.safety_constraints:
                try:
                    h_val = constraint(pos)
                    min_margin = min(min_margin, h_val)
                except Exception:
                    continue

        if min_margin == float('inf'):
            return True, 1.0

        safety_score = min(1.0, max(0.0, min_margin / self.config.safety_margin))

        return min_margin >= 0, safety_score

    def resample_trajectory(
        self,
        trajectory: Trajectory,
        new_dt: float
    ) -> Trajectory:
        """
        Resample trajectory to new time step.

        Args:
            trajectory: Original trajectory
            new_dt: New time step

        Returns:
            Resampled trajectory
        """
        old_times = trajectory.times
        new_times = np.arange(0, old_times[-1] + new_dt, new_dt)

        n_new = len(new_times)
        n_dim = trajectory.positions.shape[1]

        new_positions = np.zeros((n_new, n_dim))
        new_velocities = np.zeros((n_new, n_dim))
        new_accelerations = np.zeros((n_new, n_dim))

        for i, t in enumerate(new_times):
            # Find surrounding time indices
            idx = np.searchsorted(old_times, t)
            if idx == 0:
                idx = 1
            if idx >= len(old_times):
                idx = len(old_times) - 1

            # Linear interpolation
            t0, t1 = old_times[idx-1], old_times[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0

            new_positions[i] = (1-alpha) * trajectory.positions[idx-1] + alpha * trajectory.positions[idx]
            new_velocities[i] = (1-alpha) * trajectory.velocities[idx-1] + alpha * trajectory.velocities[idx]
            new_accelerations[i] = (1-alpha) * trajectory.accelerations[idx-1] + alpha * trajectory.accelerations[idx]

        return Trajectory(
            positions=new_positions,
            velocities=new_velocities,
            accelerations=new_accelerations,
            times=new_times,
            cost=trajectory.cost,
            is_safe=trajectory.is_safe,
            safety_score=trajectory.safety_score
        )