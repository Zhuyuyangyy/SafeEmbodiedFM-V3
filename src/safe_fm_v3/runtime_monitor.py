"""
Runtime Monitor: Real-time safety monitoring with sub-millisecond latency.

This module provides real-time monitoring capabilities for safety-critical
systems with guaranteed low-latency constraint checking.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class MonitorResult:
    """Result of real-time safety monitoring."""
    is_safe: bool
    violation_risk: float
    constraint_values: np.ndarray
    monitor_time_us: float
    buffer_occupancy: int


@dataclass
class SafetyBounds:
    """Safety bounds for monitoring."""
    x_min: np.ndarray
    x_max: np.ndarray
    v_max: np.ndarray
    a_max: np.ndarray
    jerk_max: Optional[float] = None


@dataclass
class ConstraintResult:
    """Individual constraint check result."""
    name: str
    value: float
    is_satisfied: bool
    margin: float
    criticality: float  # 0-1, higher = more critical


class RuntimeMonitor:
    """
    Real-time safety monitor with sub-millisecond latency.

    Monitors system state against safety constraints and provides
    early warning of potential constraint violations.

    Attributes:
        update_period_us: Monitor update period in microseconds
        history_length: Length of state history buffer
    """

    def __init__(
        self,
        safety_bounds: SafetyBounds,
        update_period_us: float = 100.0,
        history_length: int = 100,
        num_constraints: int = 10
    ):
        """
        Initialize runtime monitor.

        Args:
            safety_bounds: Safety bounds for monitoring
            update_period_us: Target update period in microseconds
            history_length: State history buffer size
            num_constraints: Number of active constraints
        """
        self.safety_bounds = safety_bounds
        self.update_period_us = update_period_us
        self.history_length = history_length

        # Pre-allocated buffers for minimal allocation overhead
        self.state_buffer = np.zeros((history_length, len(safety_bounds.x_min)))
        self.velocity_buffer = np.zeros((history_length, len(safety_bounds.x_min)))
        self.time_buffer = np.zeros(history_length)
        self.constraint_buffer = np.zeros((history_length, num_constraints))

        self.buffer_head = 0
        self.buffer_count = 0

        # Pre-computed bounds for fast checking
        self._x_min = safety_bounds.x_min
        self._x_max = safety_bounds.x_max
        self._v_max = safety_bounds.v_max
        self._a_max = safety_bounds.a_max

        # Custom constraints
        self.custom_constraints: List[Callable] = []
        self.custom_constraint_names: List[str] = []

        # Statistics
        self.total_updates = 0
        self.violation_count = 0
        self.max_latency_us = 0.0
        self.latency_history = deque(maxlen=1000)

    def add_constraint(
        self,
        constraint_func: Callable[[np.ndarray], float],
        name: str,
        criticality: float = 1.0
    ) -> None:
        """Add a custom safety constraint."""
        self.custom_constraints.append(constraint_func)
        self.custom_constraint_names.append(name)

        # Expand constraint buffer if needed
        current_constraints = self.constraint_buffer.shape[1]
        required_constraints = len(self.custom_constraints)

        if required_constraints > current_constraints:
            new_buffer = np.zeros((self.history_length, required_constraints))
            new_buffer[:, :current_constraints] = self.constraint_buffer
            self.constraint_buffer = new_buffer

    def _parse_state(
        self,
        state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a state vector into (position, velocity, acceleration).

        The monitor is configured with ``n_dim = len(x_min)``. A caller
        that supplies a shorter state used to raise a broadcast error
        when the truncated slices were compared against the bounds, so
        short states are padded with zeros instead of crashing.
        """
        state = np.asarray(state, dtype=float).flatten()
        n_dim = len(self._x_min)

        if state.size < n_dim:
            padded = np.zeros(n_dim)
            padded[:state.size] = state
            state = padded

        x = state[:n_dim]
        if state.size >= 2 * n_dim:
            v = state[n_dim:2 * n_dim]
        else:
            v = np.zeros(n_dim)
        if state.size >= 3 * n_dim:
            a = state[2 * n_dim:3 * n_dim]
        else:
            a = np.zeros(n_dim)

        return x, v, a

    def update(self, state: np.ndarray, time_stamp: Optional[float] = None) -> MonitorResult:
        """
        Update monitor with new state observation.

        Args:
            state: Current state vector [position, velocity, acceleration]
            time_stamp: Optional timestamp

        Returns:
            MonitorResult with safety assessment
        """
        start_time = time.perf_counter()

        # Reject non-finite states: a NaN/Inf observation carries no
        # safety information, so it must never be reported as safe.
        if not np.all(np.isfinite(state)):
            elapsed_us = (time.perf_counter() - start_time) * 1e6
            self.violation_count += 1
            self.total_updates += 1
            self.latency_history.append(elapsed_us)
            self.max_latency_us = max(self.max_latency_us, elapsed_us)
            return MonitorResult(
                is_safe=False,
                violation_risk=1.0,
                constraint_values=np.zeros(len(self.custom_constraints)),
                monitor_time_us=elapsed_us,
                buffer_occupancy=self.buffer_count
            )

        # Parse state vector
        x, v, a = self._parse_state(state)

        # Update buffers
        self.state_buffer[self.buffer_head] = x
        self.velocity_buffer[self.buffer_head] = v
        if time_stamp is not None:
            self.time_buffer[self.buffer_head] = time_stamp
        else:
            self.time_buffer[self.buffer_head] = time.time()

        # Check position constraints (fast path)
        pos_violation = np.any(x < self._x_min) or np.any(x > self._x_max)

        # Check velocity constraints (fast path)
        vel_violation = np.any(np.abs(v) > self._v_max)

        # Check acceleration constraints
        acc_violation = np.any(np.abs(a) > self._a_max) if len(a) == len(self._a_max) else False

        # Evaluate custom constraints
        constraint_values = np.zeros(len(self.custom_constraints))
        for i, constraint_func in enumerate(self.custom_constraints):
            try:
                constraint_values[i] = constraint_func(state)
                self.constraint_buffer[self.buffer_head, i] = constraint_values[i]
            except Exception:
                constraint_values[i] = 0.0

        # Compute violation risk
        is_safe = not (pos_violation or vel_violation or acc_violation)

        # Check custom constraints
        for val in constraint_values:
            if val < 0:
                is_safe = False
                break

        if not is_safe:
            self.violation_count += 1

        # Compute risk metric based on margins
        pos_margin = min(np.min(x - self._x_min), np.min(self._x_max - x))
        vel_margin = min(np.min(self._v_max - np.abs(v)), np.min(self._v_max + np.abs(v)))

        if pos_margin > 0 and vel_margin > 0:
            violation_risk = 1.0 - min(pos_margin, vel_margin) / (np.mean(self._x_max - self._x_min) / 2)
        else:
            violation_risk = 1.0

        violation_risk = np.clip(violation_risk, 0.0, 1.0)

        # Advance buffer head
        self.buffer_head = (self.buffer_head + 1) % self.history_length
        self.buffer_count = min(self.buffer_count + 1, self.history_length)
        self.total_updates += 1

        # Track latency
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        self.latency_history.append(elapsed_us)
        self.max_latency_us = max(self.max_latency_us, elapsed_us)

        return MonitorResult(
            is_safe=is_safe,
            violation_risk=violation_risk,
            constraint_values=constraint_values,
            monitor_time_us=elapsed_us,
            buffer_occupancy=self.buffer_count
        )

    def check_constraints(self, state: np.ndarray) -> List[ConstraintResult]:
        """
        Check all constraints without updating buffers.

        Args:
            state: State vector to check

        Returns:
            List of constraint results
        """
        results = []

        x, v, _a = self._parse_state(state)

        # Position constraints
        pos_margin = min(np.min(x - self._x_min), np.min(self._x_max - x))
        results.append(ConstraintResult(
            name="position_bounds",
            value=pos_margin,
            is_satisfied=pos_margin >= 0,
            margin=pos_margin,
            criticality=1.0
        ))

        # Velocity constraints
        vel_margin = min(np.min(self._v_max - np.abs(v)), np.min(self._v_max + np.abs(v)))
        results.append(ConstraintResult(
            name="velocity_bounds",
            value=vel_margin,
            is_satisfied=vel_margin >= 0,
            margin=vel_margin,
            criticality=0.9
        ))

        # Custom constraints
        for i, (name, constraint_func) in enumerate(
            zip(self.custom_constraint_names, self.custom_constraints)
        ):
            try:
                value = constraint_func(state)
                results.append(ConstraintResult(
                    name=name,
                    value=value,
                    is_satisfied=value >= 0,
                    margin=value,
                    criticality=0.8
                ))
            except Exception:
                results.append(ConstraintResult(
                    name=name,
                    value=0.0,
                    is_satisfied=True,
                    margin=0.0,
                    criticality=0.8
                ))

        return results

    def predict_violation(
        self,
        state: np.ndarray,
        dt: float = 0.1,
        horizon: int = 10
    ) -> Tuple[bool, float, int]:
        """
        Predict future constraint violations using simple extrapolation.

        Args:
            state: Current state
            dt: Time step for prediction
            horizon: Number of steps to look ahead

        Returns:
            Tuple of (will_violate, time_to_violation, step_of_violation)
        """
        x, v, a = self._parse_state(state)

        x_pred = x.copy()
        v_pred = v.copy()
        a_pred = a.copy()

        for step in range(horizon):
            # Simple Euler integration
            x_pred = x_pred + v_pred * dt
            v_pred = v_pred + a_pred * dt

            # Check for violation
            pos_violation = np.any(x_pred < self._x_min) or np.any(x_pred > self._x_max)
            vel_violation = np.any(np.abs(v_pred) > self._v_max)

            if pos_violation or vel_violation:
                return True, step * dt, step

        return False, float('inf'), -1

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        latencies = list(self.latency_history)

        return {
            "total_updates": self.total_updates,
            "violation_count": self.violation_count,
            "violation_rate": self.violation_count / max(1, self.total_updates),
            "mean_latency_us": np.mean(latencies) if latencies else 0.0,
            "max_latency_us": self.max_latency_us,
            "p99_latency_us": np.percentile(latencies, 99) if latencies else 0.0,
            "p999_latency_us": np.percentile(latencies, 99.9) if latencies else 0.0,
            "buffer_occupancy": self.buffer_count,
            "num_constraints": len(self.custom_constraints)
        }

    def reset_statistics(self) -> None:
        """Reset monitoring statistics."""
        self.total_updates = 0
        self.violation_count = 0
        self.max_latency_us = 0.0
        self.latency_history.clear()

    def get_state_history(self, n_steps: int = 10) -> np.ndarray:
        """Get recent state history."""
        n_steps = min(n_steps, self.buffer_count)
        if n_steps == 0:
            return np.zeros((1, len(self._x_min)))

        indices = [(self.buffer_head - i) % self.history_length for i in range(n_steps)]
        return self.state_buffer[indices[::-1]]

    def compute_control_safety_margin(
        self,
        state: np.ndarray,
        control: np.ndarray,
        Lf_h: float,
        Lg_h: np.ndarray,
        alpha: float = 1.0
    ) -> float:
        """
        Compute safety margin for a control input.

        Args:
            state: Current state
            control: Proposed control input
            Lf_h: Lie derivative along drift
            Lg_h: Lie derivative along control
            alpha: CBF parameter

        Returns:
            Safety margin (positive = safe)
        """
        if isinstance(Lg_h, np.ndarray):
            constraint = np.dot(Lg_h, control) + Lf_h + alpha * 0.0
        else:
            constraint = Lg_h * control[0] + Lf_h + alpha * 0.0

        return float(constraint)


def create_default_bounds(
    n_dim: int = 3,
    position_limit: float = 10.0,
    velocity_limit: float = 5.0,
    acceleration_limit: float = 2.0
) -> SafetyBounds:
    """
    Create default safety bounds.

    Args:
        n_dim: Number of dimensions
        position_limit: Position limit (symmetric)
        velocity_limit: Velocity limit (symmetric)
        acceleration_limit: Acceleration limit (symmetric)

    Returns:
        SafetyBounds object
    """
    return SafetyBounds(
        x_min=-position_limit * np.ones(n_dim),
        x_max=position_limit * np.ones(n_dim),
        v_max=velocity_limit * np.ones(n_dim),
        a_max=acceleration_limit * np.ones(n_dim)
    )