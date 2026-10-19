# SafeEmbodiedFM-V3

Safe Embodied **F**oundation **M**odel V3 - Formal safety verification and control barrier functions for embodied AI systems.

## What "FM" means here

In this project **"FM" stands for Foundation Model** -- a large pretrained
model (typically an LLM or vision-language model) that is being adapted
to, or interfaced with, physical embodied agents such as mobile robots,
manipulators and autonomous vehicles. The V3 release adds a dedicated
formal-safety layer that wraps any foundation-model-driven controller:

* the foundation model supplies *nominal* control or planning intent;
* this library verifies and/or modifies the nominal output through
  Control Barrier Functions, Lyapunov functions, runtime monitors and
  collision detection so the foundation model can be used in
  safety-critical settings without re-training the underlying model.

In short: **Safe + Embodied + Foundation Model = a safety filter for
foundation-model-driven embodied agents**.

## Overview

SafeEmbodiedFM-V3 provides a comprehensive toolkit for ensuring safety in embodied AI systems, including robotic manipulators, autonomous vehicles, and other physical agents. The library combines formal verification methods with real-time safety monitoring to guarantee system safety under all operating conditions.

## Features

- **Formal Lyapunov Verification**: Verify stability of dynamical systems using formal Lyapunov functions
- **Control Barrier Function (CBF) Synthesis**: Automatically synthesize CBFs from safety specifications
- **CVXPY-based CBF-QP Solver (R2)**: Exact quadratic-programming reference implementation of the CBF safety filter
- **Real-Time Safety Monitoring**: Sub-millisecond latency monitoring for safety-critical applications
- **Safe Trajectory Planning**: Plan trajectories that satisfy safety constraints
- **Point Cloud Collision Detection**: Fast collision detection using spatial hashing

## R2: CVXPY-based CBF QP Solver

In addition to the fast projected-gradient CBF solver in
`cbf_synthesizer.compute_safe_control`, R2 adds a reference QP solver
backed by CVXPY. Use it for offline analysis, hardware-in-the-loop
ground-truth, or whenever you need the numerically optimal safe input.

```python
import numpy as np
from safe_fm_v3 import solve_cbf_qp

u_nominal = np.array([0.0, 0.0, 0.0])
Lg_h = np.array([1.0, 0.0, 0.0])  # control Lie derivative
Lf_h = 0.0                          # drift Lie derivative
h_value = 1.0                       # current barrier value
result = solve_cbf_qp(
    u_nominal=u_nominal,
    Lg_h=Lg_h, Lf_h=Lf_h, h_value=h_value,
    alpha=1.0, slack=0.1,
)
print("status:", result.solver_status, "u:", result.u,
      "time_ms:", round(result.solve_time_ms, 3))
```

CVXPY is an *optional* dependency. Install it with `pip install cvxpy`.
If CVXPY is not installed, `is_qp_available()` returns `False` and
`solve_cbf_qp()` raises `CBFQPUnavailable`; the rest of the library
remains functional.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Lyapunov Stability Verification

```python
import numpy as np
from safe_fm_v3 import LyapunovVerifier

# Define Lyapunov candidate V(x) = x^T P x
P = np.eye(3)

def V(x): return float(x.T @ P @ x)
def grad_V(x): return 2 * P @ x
def f(x, u): return -x  # Stable dynamics

verifier = LyapunovVerifier()
result = verifier.verify_lyapunov_function(V, grad_V, f, domain_radius=10.0)

print(f"Stability verified: {result.is_stable}")
print(f"Region of attraction: {result.roa_estimate}")
```

### CBF-Based Safe Control

```python
import numpy as np
from safe_fm_v3 import CBFSynthesizer

synthesizer = CBFSynthesizer(alpha=1.0)

# Define safety constraint h(x) >= 0
def h(x): return 1.0 - np.linalg.norm(x)
def grad_h(x): return -x / (np.linalg.norm(x) + 1e-6)

# Compute safe control input
u_nominal = np.array([0.1, 0.1, 0.1])
result = synthesizer.compute_safe_control(h, grad_h, Lf_h, Lg_h, u_nominal)

print(f"Safe control: {result.u}")
print(f"Constraint satisfied: {result.constraint_satisfaction}")
```

### Real-Time Monitoring

```python
import numpy as np
from safe_fm_v3 import RuntimeMonitor, SafetyBounds

# Define safety bounds
bounds = SafetyBounds(
    x_min=-5 * np.ones(3),
    x_max=5 * np.ones(3),
    v_max=2 * np.ones(3),
    a_max=1 * np.ones(3)
)

monitor = RuntimeMonitor(bounds)

# Monitor system state
state = np.array([1.0, 0.5, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
result = monitor.update(state)

print(f"Is safe: {result.is_safe}")
print(f"Latency: {result.monitor_time_us:.2f}us")
```

### Safe Trajectory Planning

```python
import numpy as np
from safe_fm_v3 import SafePlanner, PlannerConfig, PlannerType

config = PlannerConfig(planner_type=PlannerType.MPCC, horizon=100)
planner = SafePlanner(config)

# Plan safe trajectory
start = np.array([0.0, 0.0, 0.0])
goal = np.array([1.0, 1.0, 1.0])
trajectory = planner.plan(start, goal)

print(f"Trajectory cost: {trajectory.cost}")
print(f"Safety score: {trajectory.safety_score}")
```

### Collision Detection

```python
import numpy as np
from safe_fm_v3 import CollisionDetector

detector = CollisionDetector()

# Add obstacles
detector.add_sphere_obstacle(np.array([1.0, 0.0, 0.0]), 0.5)
detector.add_box_obstacle(np.array([2.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3]))

# Check collision
result = detector.check_collision(np.array([1.0, 0.0, 0.0]), radius=0.1)

print(f"Collision: {result.is_collision}")
print(f"Distance: {result.distance:.3f}")
```

## Components

### LyapunovVerifier

Formal verification of Lyapunov stability for nonlinear dynamical systems.

```python
verifier = LyapunovVerifier(
    tolerance=1e-6,
    grid_resolution=50,
    max_iterations=1000
)
```

### CBFSynthesizer

Synthesis of control barrier functions from safety specifications.

```python
synthesizer = CBFSynthesizer(
    relative_degree=1,
    alpha=1.0,
    epsilon=1e-6
)
```

### RuntimeMonitor

Real-time safety monitoring with sub-millisecond latency.

```python
monitor = RuntimeMonitor(
    safety_bounds=bounds,
    update_period_us=100.0,
    history_length=100
)
```

### SafePlanner

Constrained trajectory planning with safety guarantees.

```python
planner = SafePlanner(
    config=PlannerConfig(
        planner_type=PlannerType.MPCC,
        horizon=100,
        max_velocity=5.0
    ),
    safety_constraints=[constraint1, constraint2]
)
```

### CollisionDetector

Fast collision detection using spatial hashing.

```python
detector = CollisionDetector(
    checker_type=CollisionType.POINT_SPHERE,
    spatial_resolution=0.1
)
```

## Architecture

```
safe_fm_v3/
├── __init__.py              # Package initialization
├── lyapunov_verifier.py    # Lyapunov stability verification
├── cbf_synthesizer.py       # Control barrier function synthesis
├── qp_solver.py             # CVXPY-based CBF-QP reference solver (R2)
├── runtime_monitor.py # Real-time safety monitoring
├── safe_planner.py          # Constrained trajectory planning
└── collision_detector.py    # Point cloud collision detection
```

## Testing

Run the test suite:

```bash
pytest tests/test_safe.py -v
```

Run the R2 QP solver tests:

```bash
pytest tests/test_qp_solver.py -v
```

Run specific test classes:

```bash
pytest tests/test_safe.py::TestLyapunovVerifier -v
pytest tests/test_safe.py::TestCBFSynthesizer -v
pytest tests/test_safe.py::TestRuntimeMonitor -v
```

## Requirements

- Python 3.8+
- NumPy

## Performance

- **Runtime Monitor**: < 1ms latency (typical ~100-500us)
- **Collision Detection**: < 100us per check
- **CBF Synthesis**: < 100ms
- **Lyapunov Verification**: < 5s for full verification

## License

MIT License

## References

Full citations for the methods implemented here are listed in [CITATIONS.md](CITATIONS.md).

- [Control Barrier Functions: Theory and Applications](link)
- [Control Lyapunov Function](link)
- [Control Barrier Function Quadratic Programming](link)