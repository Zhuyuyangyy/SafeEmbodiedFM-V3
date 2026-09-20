# SafeEmbodiedFM-V3 Q2 Review (7-dim rubric)

Scope: `src/safe_fm_v3/` — `safe_planner.py` (601 lines), `cbf_synthesizer.py`
(405 lines), `runtime_monitor.py` (405 lines), plus `safe_planner`,
`lyapunov_verifier`, `collision_detector`. Pure-numpy safety stack:
CBF synthesis, trajectory planning, runtime monitoring.

## 1. Correctness
- `cbf_synthesizer.synthesize_from_spec` is mostly a wrapper that returns
  the input constraint as the barrier; `is_valid_cbf=True` is set
  unconditionally. `_compute_lie_relative_degree` tests only `Lf_h`
  magnitude at one point and falls back to the configured
  `relative_degree` — that does not prove the function is a valid CBF.
- `cbf_synthesizer.compute_safe_control` solves the QP with a single
  gradient-projection step (`u + step * Lg_h`) — not a true QP solver,
  so the returned `u_safe` does not in general satisfy
  `Lg_h u >= -Lf_h + alpha h`; only a sufficient improvement. The
  class is named "QP" in the docstring but the body is projected
  gradient, which is a different guarantee.
- `safe_planner._plan_ilqr` builds a feedback gain `K` as zeros and
  updates positions only inside the loop; the "backward pass" is
  effectively a feedforward correction, so this is not iLQR. The cost
  reported does not reflect real state-tracking cost.
- `runtime_monitor.update` indexes `state[2*n_dim:3*n_dim]` for
  acceleration without verifying length; the fallback path
  (`v = np.zeros(n_dim)`) masks malformed inputs. `violation_risk`
  formula uses `np.mean(x_max - x_min)/2` as the denominator — a
  constant scale, not a state-dependent margin, so risk is not
  normalized per scenario.
- `create_box_cbf.grad_h` uses `if min_dist in dist_to_min:` — float
  `in` is a numpy elementwise check that returns an array, which is
  always truthy, so the gradient branch is effectively random.
  This is a real bug in a safety primitive.

## 2. Architecture / Modularity
- Clean separation: planner (trajectory), synthesizer (CBF), monitor
  (runtime). Dataclass configs (`PlannerConfig`, `SafetyBounds`) keep
  tunables together. `Waypoint`, `Trajectory`, `CBFResult`,
  `ControlInput` are explicit return types — good for readability.
- `SafePlanner` is a god-class: it implements Hermite, MPCC, RRT, and
  iLQR. Splitting into per-planner classes would improve testability.
  `runtime_monitor` is well-scoped: a single monitor, multiple
  constraints, clear stats surface.

## 3. Code Quality
- Type hints on public APIs (`np.ndarray`, `Callable`, `Optional`,
  `Tuple`). `Enum` for planner / CBF types — strong. `dataclass`
  used everywhere.
- Several `try/except Exception: continue` blocks in safety-critical
  paths (constraint checks) — these will silently treat an exception
  as a safe point. For a safety module this is a code smell: a
  failing constraint function should fail loud, not be ignored.
  Comments sometimes contradict code (e.g. "Simple gradient-based
  projection" while the docstring says QP).

## 4. Documentation
- Each module/class has a descriptive top docstring. Methods have
  short summaries; Args/Returns are present. `create_circle_cbf` /
  `create_box_cbf` factory docstrings are clear and copy-pasteable.
- Missing: any reference to the CBF validity conditions, the
  algorithm-vs-implementation gap (QP vs. projected gradient), and
  the iLQR implementation caveat. A user reading the docstring of
  `compute_safe_control` would expect a QP solve, not what is
  actually done.

## 5. Testing
- `tests/` dir exists at project root; the read scope didn't show
  specific test files. The `create_circle_cbf` / `create_box_cbf`
  factories are easy to unit-test against known gradients
  (analytical: `-2(x - c)` for circle, `+/- e_i` for box). The
  `grad_h` bug in `create_box_cbf` is exactly the kind of test
  a numeric-gradient sanity check would catch. `runtime_monitor`
  edge cases (malformed state length) are not visibly tested.

## 6. Performance
- `runtime_monitor` pre-allocates buffers and avoids per-update
  allocation — good. `safe_planner._plan_rrt` is O(N) per iteration
  for nearest-neighbor (linear scan), fine for a 100-iter budget.
  `_plan_ilqr` rebuilds the trajectory from Hermite every outer
  iter — wasteful. `cbf_synthesizer` is O(1) per call; not a
  bottleneck.

## 7. Security / Safety
- This module is the safety layer — and the safety primitives have
  the most concerning correctness gaps: `compute_safe_control` is
  not a true QP, `synthesize_from_spec` returns `is_valid_cbf=True`
  without checking, `create_box_cbf.grad_h` has a logic bug that
  makes the gradient effectively undefined for non-degenerate
  boxes, and exceptions in constraint functions are swallowed
  silently. For a research prototype these are acceptable; for any
  real-world deployment each is a fail-unsafe hazard.
- The runtime monitor is the most trustworthy piece: it has clear
  bounds, latency tracking, p99/p999 stats, and a separate
  `predict_violation` for early warning. Good pattern.

## Summary
Architecturally clean: planner / synthesizer / monitor with explicit
dataclasses and enums. Runtime monitor is the strongest module.
Planners are crammed into one class, and the safety primitives
(CBF synthesis, CBF-QP, box CBF gradient) have correctness gaps that
matter — including a real bug in `create_box_cbf.grad_h` and a
projected-gradient implementation behind a "QP" docstring. The
silent exception handling in constraint checks is the wrong default
for a safety module. Useful as a research baseline; would need
formal CBF validation, a real QP solver, and tighter unit tests
before any safety claim.
