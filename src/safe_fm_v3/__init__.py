"""
SafeEmbodiedFM-V3: Safe Embodied Foundation Model V3
=====================================================

**What "FM" means in this project**

In ``SafeEmbodiedFM-V3`` the suffix **"FM"** stands for **Foundation Model** --
in particular, a large pretrained model that is being adapted to (or
interfaced with) physical, embodied agents such as mobile robots,
manipulators and autonomous vehicles. The V3 release adds a dedicated
formal-safety layer that wraps any foundation-model-driven controller:

  * the foundation model supplies *nominal* control or planning intent;
  * the safety layer (this library) verifies and/or modifies the
    nominal output to guarantee constraint satisfaction.

The safety layer is implemented via **Control Barrier Functions (CBFs)**,
**Lyapunov functions**, **runtime monitoring** and **collision detection**
so that the foundation model can be used in safety-critical settings
without re-training the underlying model.

Core components:
- Lyapunov verifier for formal stability analysis
- CBF synthesizer for safety-critical control
- CVXPY-based QP solver for the CBF safety filter
- Runtime monitor for real-time safety monitoring
- Safe planner for constrained trajectory optimization
- Collision detector for point cloud based collision checking
"""

__version__ = "3.0.0"
__author__ = "SafeEmbodiedFM Team"

from .lyapunov_verifier import LyapunovVerifier
from .cbf_synthesizer import CBFSynthesizer
from .runtime_monitor import RuntimeMonitor, SafetyBounds
from .safe_planner import SafePlanner, PlannerConfig, PlannerType
from .collision_detector import CollisionDetector
from .qp_solver import (
    CBFQPResult,
    CBFQPUnavailable,
    is_qp_available,
    solve_cbf_qp,
)

__all__ = [
    "LyapunovVerifier",
    "CBFSynthesizer",
    "RuntimeMonitor",
    "SafePlanner",
    "CollisionDetector",
    "SafetyBounds",
    "PlannerConfig",
    "PlannerType",
    # New in R2: CVXPY-based CBF QP solver
    "solve_cbf_qp",
    "CBFQPResult",
    "CBFQPUnavailable",
    "is_qp_available",
    "__version__",
]