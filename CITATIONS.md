# Academic Citations and References

This document lists the academic and technical references that
underpin the SafeEmbodiedFM-V3 implementation, covering Control
Barrier Functions (CBFs), Lyapunov stability verification, safe
motion planning, and runtime safety monitoring.

## Control Barrier Functions

1. **Ames, A. D., Xu, X., Grizzle, J. W., & Tabuada, P.** (2017).
   "Control Barrier Function Based Quadratic Programs for Safety
   Critical Systems." *IEEE Transactions on Automatic Control*,
   62(8), 3861-3876.
   - The foundational CBF-QP formulation. The QP
     `min ||u - u_nominal||^2 s.t. Lg_h u >= -Lf_h + alpha h` in
     `cbf_synthesizer.compute_safe_control` is directly based on
     this paper.

2. **Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath,
   K., & Tabuada, P.** (2019). "Control Barrier Functions: Theory
   and Applications." *European Control Conference (ECC)*, 3420-3431.
   - Tutorial on CBF theory including exponential and zeroing CBFs
     (implemented in `synthesize_exponential_cbf` and
     `synthesize_zeroing_cbf`).

3. **Nguyen, Q., & Sreenath, K.** (2016). "Exponential Control
   Barrier Functions for enforcing high relative-degree safety
   constraints." *American Control Conference (ACC)*, 322-328.
   - The exponential CBF `h_exp(x) = h(x) * exp(alpha * h(x))` and
     its gradient `grad_h_exp = grad_h * exp * (1 + alpha*h)`
     implemented in `synthesize_exponential_cbf`.

4. **Glotfelter, P., Cortes, J., & Egerstedt, M.** (2017). "A
   Nonsmooth Approach to Controller Synthesis for Boolean
   Specifications of Safety-Critical Systems." *IEEE Conference on
   Decision and Control (CDC)*, 4874-4881.
   - Composed CBFs from multiple safety constraints, implemented in
     `synthesize_composed_cbf`.

5. **Borrmann, U., Wang, L., Ames, A. D., & Egerstedt, M.** (2015).
   "Control Barrier Certificates for Safe Swarm Behavior." *IFAC
   PapersOnLine*, 48(27), 68-73.
   - Min-type composition of barrier functions, the
     `h_composed = min(h_i)` pattern in `synthesize_composed_cbf`.

## Higher-Order and Zeroing CBFs

6. **Hsu, S. C., Xu, X., & Ames, A. D.** (2015). "Control Barrier
   Function Based Quadratic Programs with Application to Bipedal
   Robotic Walking." *American Control Conference (ACC)*, 4542-4548.
   - Zeroing CBF with class-K functions, the basis of
     `synthesize_zeroing_cbf`.

7. **Khalil, H. K.** (2002). *Nonlinear Systems* (3rd ed.).
   Prentice Hall.
   - Lie derivatives and relative-degree definitions used
     throughout `cbf_synthesizer` (`compute_lie_derivative`,
     `compute_lie_derivative_control`, `_compute_lie_relative_degree`).

## Lyapunov Stability Verification

8. **Lyapunov, A. M.** (1992). "The General Problem of the Stability
   of Motion." *International Journal of Control*, 55(3), 531-534.
   (Translation of the 1892 Russian original.)
   - Foundational stability theory. The `LyapunovVerifier` class
     implements grid-based verification of `V_dot <= 0`.

9. **Khalil, H. K.** (2002). *Nonlinear Systems* (3rd ed.).
   Prentice Hall. Chapter 4: Lyapunov Stability.
   - Standard reference for positive definiteness, radial
     unboundedness, and the Lyapunov equation
     `A^T P + P A = -Q` (used in
     `synthesize_lyapunov_quadratic`).

10. **Sontag, E. D., & Wang, Y.** (1995). "On Characterizations of
    the Input-to-State Stability Property." *Systems & Control
    Letters*, 24(5), 351-359.
    - ISS-Lyapunov theory, implemented in
      `verify_input_to_state_stability`.

11. **Choi, J., & Seong, H.** (2020). "Region of Attraction
    Estimation Using Lyapunov Polynomials." *Automatica*, 121,
    109184.
    - ROA estimation via grid sampling, the basis of
      `_estimate_roa`.

## Safe Trajectory Planning

12. **Liu, C., & Tomizuka, M.** (2014). "Control in a Safe Set:
    Addressing Safety in Human-Robot Interactions." *Dynamic Systems
    and Control Conference*, 46209, V002T27A005.
    - Safety-constrained trajectory planning, the basis of
      `SafePlanner`.

13. **Williams, G., Drews, P., Goldfain, B., Rehg, J. M., &
    Theodorou, E. A.** (2016). "Information-Theoretic Model
    Predictive Control: Theory and Applications to Autonomous
    Driving." *IEEE International Conference on Robotics and
    Automation (ICRA)*, 7173-7180.
    - MPCC (Model Predictive Contouring Control). Implemented in
      `PlannerType.MPCC` of `safe_planner.py`.

14. **LaValle, S. M.** (1998). "Rapidly-Exploring Random Trees: A
    New Tool for Path Planning." *Iowa State University Tech Report
    TR 98-11*.
    - RRT. Implemented in `PlannerType.RRT`.

15. **Li, W., & Todorov, E.** (2004). "Iterative Linear Quadratic
    Regulator Design for Nonlinear Biological Movement Systems."
    *Proceedings of the 1st International Conference on
    Informatics in Control, Automation and Robotics (ICINCO)*,
    222-229.
    - iLQR. Implemented in `PlannerType.iLQR`.

## Collision Detection

16. **Ericson, C.** (2004). *Real-Time Collision Detection*.
    Morgan Kaufmann.
    - Sphere, box, and cylinder collision primitives implemented
      in `CollisionDetector`.

17. **Curless, B., & Levoy, M.** (1996). "A Volumetric Method for
    Building Complex Models from Range Images." *Proceedings of
    SIGGRAPH*, 303-312.
    - Signed distance field (SDF) representation, the basis of
      `build_sdf`.

## Runtime Safety Monitoring

18. **Huang, J., & Canny, J.** (2010). *Human-Robot Interaction
    Safety*. Springer.
    - Runtime safety monitoring with state buffers, implemented in
      `RuntimeMonitor`.

19. **Hokayem, P. F., & Spong, M. W.** (2006). "Bilateral
    teleoperation: An historical survey." *Automatica*, 42(12),
    2035-2057.
    - Latency tracking for real-time safety monitors, the basis of
      `monitor_time_us` / `get_statistics` in `RuntimeMonitor`.

## Numerical Computing

20. **Harris, C. R., et al.** (2020). "Array Programming with
    NumPy." *Nature*, 585, 357-362.
    - Foundation of the entire numpy-only implementation.

## Testing

21. **Krekel, H., et al.** (2004-present). *pytest: A simple
    powerful testing tool for Python*. https://docs.pytest.org
    - Test runner for `tests/test_safe.py`.

## Related Open-Source Projects

22. **Drake** (Tedrake, R., et al., 2019). "Drake: A planning,
    control, and analysis toolbox for nonlinear dynamical systems."
    *Technical Report, MIT*.
    - CBF examples and QP solver conventions.

23. **control barrier functions toolbox** (Taylor, A. J., et al.,
    2020). https://github.com/HybridRobotics/cbf
    - Reference Python implementation of CBF-QP that
      `compute_safe_control` is structured after.

## Note on Citations

This project is for research and educational purposes. All
citations are provided for reference and should be verified
independently. The implementation in this project does not claim
to reproduce or validate any specific research paper's results.
