"""
Collision Detector: Point cloud based collision detection.

This module provides fast collision detection for robotic systems using
point cloud representations and efficient spatial data structures.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time


class CollisionType(Enum):
    """Types of collision checks."""
    POINT_BOX = "point_box"
    POINT_SPHERE = "point_sphere"
    SPHERE_BOX = "sphere_box"
    SPHERE_SPHERE = "sphere_sphere"
    SPHERE_CYLINDER = "sphere_cylinder"
    POINT_CLOUD_SDF = "point_cloud_sdf"


@dataclass
class CollisionResult:
    """Result of collision detection."""
    is_collision: bool
    distance: float
    closest_point: Optional[np.ndarray]
    collision_normal: Optional[np.ndarray]
    check_time_us: float


@dataclass
class BoundingVolume:
    """Bounding volume for collision checking."""
    center: np.ndarray
    half_extents: np.ndarray
    rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix


@dataclass
class Sphere:
    """Sphere primitive for collision detection."""
    center: np.ndarray
    radius: float


@dataclass
class Cylinder:
    """Cylinder primitive for collision detection."""
    axis: np.ndarray  # Unit vector
    center: np.ndarray  # Center of base
    radius: float
    height: float


class CollisionDetector:
    """
    Fast collision detector for point clouds and geometric primitives.

    Uses spatial partitioning and efficient algorithms for real-time
    collision detection.

    Attributes:
        checker_type: Type of collision checking to use
    """

    def __init__(
        self,
        checker_type: CollisionType = CollisionType.POINT_SPHERE,
        spatial_resolution: float = 0.1,
        max_check_radius: float = 2.0
    ):
        """
        Initialize collision detector.

        Args:
            checker_type: Type of collision checking
            spatial_resolution: Resolution for spatial hashing
            max_check_radius: Maximum distance for collision checks
        """
        self.checker_type = checker_type
        self.spatial_resolution = spatial_resolution
        self.max_check_radius = max_check_radius
        # Spatial hash grid
        self.grid = {}
        self.grid_cells = {}

        # Obstacle storage
        self.obstacles: List[Dict] = []
        self.sdf_cache: Optional[Callable] = None

    def add_sphere_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        name: str = "obstacle"
    ) -> int:
        """
        Add a spherical obstacle.

        Args:
            center: Center position
            radius: Sphere radius
            name: Obstacle name

        Returns:
            Obstacle ID
        """
        obstacle_id = len(self.obstacles)
        self.obstacles.append({
            "type": "sphere",
            "center": center,
            "radius": radius,
            "name": name,
            "id": obstacle_id
        })
        self._insert_into_grid(center, obstacle_id)
        return obstacle_id

    def add_box_obstacle(
        self,
        center: np.ndarray,
        half_extents: np.ndarray,
        rotation: Optional[np.ndarray] = None,
        name: str = "obstacle"
    ) -> int:
        """
        Add a box obstacle.

        Args:
            center: Center position
            half_extents: Half extents [hx, hy, hz]
            rotation: Optional rotation matrix
            name: Obstacle name

        Returns:
            Obstacle ID
        """
        obstacle_id = len(self.obstacles)
        self.obstacles.append({
            "type": "box",
            "center": center,
            "half_extents": half_extents,
            "rotation": rotation,
            "name": name,
            "id": obstacle_id
        })

        # Add to spatial grid using multiple points
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    point = center + np.array([dx, dy, dz]) * half_extents
                    self._insert_into_grid(point, obstacle_id)

        return obstacle_id

    def add_cylinder_obstacle(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        height: float,
        name: str = "obstacle"
    ) -> int:
        """
        Add a cylindrical obstacle.

        Args:
            center: Center position
            axis: Axis unit vector
            radius: Cylinder radius
            height: Cylinder height
            name: Obstacle name

        Returns:
            Obstacle ID
        """
        obstacle_id = len(self.obstacles)
        self.obstacles.append({
            "type": "cylinder",
            "center": center,
            "axis": axis / np.linalg.norm(axis),
            "radius": radius,
            "height": height,
            "name": name,
            "id": obstacle_id
        })

        # Insert key points into spatial grid
        self._insert_into_grid(center, obstacle_id)
        self._insert_into_grid(center + axis * height, obstacle_id)

        return obstacle_id

    def add_point_cloud(
        self,
        points: np.ndarray,
        radius: float = 0.05,
        name: str = "point_cloud"
    ) -> int:
        """
        Add a point cloud as an obstacle.

        Args:
            points: Nx3 array of points
            radius: Effective radius for each point
            name: Point cloud name

        Returns:
            Point cloud ID
        """
        pc_id = len(self.obstacles)
        self.obstacles.append({
            "type": "point_cloud",
            "points": points,
            "radius": radius,
            "name": name,
            "id": pc_id
        })

        # Insert points into spatial grid
        for point in points[::10]:  # Subsample for grid insertion
            self._insert_into_grid(point, pc_id)

        return pc_id

    def build_sdf(self, resolution: float = 0.05) -> Callable:
        """
        Build signed distance field from obstacles.

        Args:
            resolution: Grid resolution

        Returns:
            SDF function that takes a point and returns distance
        """
        # Get bounding box
        all_points = []
        for obs in self.obstacles:
            if obs["type"] == "point_cloud":
                all_points.append(obs["points"])
            elif obs["type"] == "sphere":
                all_points.append(np.array([obs["center"]]))
            elif obs["type"] == "box":
                all_points.append(np.array([obs["center"]]))
            elif obs["type"] == "cylinder":
                all_points.append(np.array([obs["center"]]))

        if all_points:
            all_points = np.vstack(all_points)
            bbox_min = np.min(all_points, axis=0) - 1
            bbox_max = np.max(all_points, axis=0) + 1
        else:
            bbox_min = np.array([-5, -5, -5])
            bbox_max = np.array([5, 5, 5])

        def sdf(x: np.ndarray) -> float:
            min_dist = float('inf')
            for obs in self.obstacles:
                if obs["type"] == "sphere":
                    dist = np.linalg.norm(x - obs["center"]) - obs["radius"]
                elif obs["type"] == "box":
                    dist = self._box_sdf(x, obs["center"], obs["half_extents"], obs.get("rotation"))
                elif obs["type"] == "cylinder":
                    dist = self._cylinder_sdf(x, obs["center"], obs["axis"], obs["radius"], obs["height"])
                elif obs["type"] == "point_cloud":
                    dist = self._point_cloud_sdf(x, obs["points"], obs["radius"])
                else:
                    dist = float('inf')

                min_dist = min(min_dist, dist)

            return min_dist

        self.sdf_cache = sdf
        return sdf

    def _point_cloud_sdf(
        self,
        x: np.ndarray,
        points: np.ndarray,
        radius: float
    ) -> float:
        """Compute SDF for point cloud."""
        # KD-tree style search (simplified)
        min_dist = float('inf')

        # Only check nearby points using spatial hash
        nearby_ids = self._get_nearby_cells(x)

        for obs in self.obstacles:
            if obs["type"] == "point_cloud" and obs["id"] in nearby_ids:
                points_obs = obs["points"]
                for p in points_obs:
                    dist = np.linalg.norm(x - p) - radius
                    min_dist = min(min_dist, dist)

                    if min_dist < 0:
                        return min_dist

        return min_dist

    def _obstacle_bound(self, obs: Dict) -> Tuple[np.ndarray, float]:
        """Return ``(center, bounding_radius)`` of an obstacle.

        The bounding sphere must enclose the whole primitive so that a
        sphere-sphere broad-phase test is conservative (never discards a
        primitive that could actually touch the query sphere).
        """
        if obs["type"] == "sphere":
            return obs["center"], obs["radius"]
        if obs["type"] == "box":
            half = np.asarray(obs["half_extents"], dtype=float)
            return obs["center"], float(np.linalg.norm(half))
        if obs["type"] == "cylinder":
            axis = obs["axis"]
            # Base center is ``center``; the top is ``center + axis*height``.
            mid = obs["center"] + 0.5 * axis * obs["height"]
            radius = obs["height"] / 2.0 + obs["radius"]
            return mid, radius
        if obs["type"] == "point_cloud":
            points = obs["points"]
            if len(points) == 0:
                return np.zeros(3), -1.0  # never matches
            center = points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(points - center, axis=1)) + obs["radius"])
            return center, radius
        return obs.get("center", np.zeros(3)), 0.0

    def _broad_phase(
        self,
        position: np.ndarray,
        radius: float,
        ignore_ids: set
    ) -> List[Dict]:
        """Vectorized bounding-sphere cull over all obstacles."""
        if not self.obstacles:
            return []

        centers = np.empty((len(self.obstacles), len(position)))
        radii = np.empty(len(self.obstacles))
        ids = np.empty(len(self.obstacles), dtype=int)

        for i, obs in enumerate(self.obstacles):
            c, r = self._obstacle_bound(obs)
            centers[i] = c
            radii[i] = r
            ids[i] = obs["id"]

        query = np.asarray(position, dtype=float)
        # Only ignore_ids can drop an obstacle; otherwise keep everything
        # whose bounding sphere comes within ``radius`` of the query.
        reach = radii + radius
        dists = np.linalg.norm(centers - query, axis=1)

        keep = (dists <= reach) & ~np.isin(ids, list(ignore_ids))
        if not keep.any():
            # No obstacle is within reach, but callers still expect the
            # nearest obstacle to be reported (distance / closest point /
            # normal), so fall back to the single closest one.
            order = np.argsort(dists)
            for i in order:
                if ids[i] not in ignore_ids:
                    keep[i] = True
                    break
        return [self.obstacles[i] for i in np.nonzero(keep)[0]]

    def check_collision(
        self,
        position: np.ndarray,
        radius: float = 0.1,
        ignore_ids: Optional[List[int]] = None
    ) -> CollisionResult:
        """
        Check collision at a position.

        Args:
            position: Position to check
            radius: Agent radius
            ignore_ids: Obstacle IDs to ignore

        Returns:
            CollisionResult
        """
        start_time = time.perf_counter()
        ignore_ids = set(ignore_ids or [])

        # Broad phase: a single vectorized bounding-sphere test over all
        # obstacles. It guarantees recall (nothing that could possibly be
        # touching the agent is dropped) and is far cheaper than walking
        # thousands of hash cells, which was the old bottleneck.
        candidates = self._broad_phase(position, radius, ignore_ids)

        min_distance = float('inf')
        closest_point = None
        collision_normal = None

        for obs in candidates:
            if obs["type"] == "sphere":
                dist, closest = self._sphere_point_distance(position, obs["center"], obs["radius"])
            elif obs["type"] == "box":
                dist, closest = self._box_point_distance(position, obs["center"], obs["half_extents"], obs.get("rotation"))
            elif obs["type"] == "cylinder":
                dist, closest = self._cylinder_point_distance(position, obs["center"], obs["axis"], obs["radius"], obs["height"])
            elif obs["type"] == "point_cloud":
                dist, closest = self._point_cloud_point_distance(position, obs["points"], obs["radius"])
            else:
                dist = float('inf')
                closest = None

            if dist < min_distance:
                min_distance = dist
                closest_point = closest

        collision_distance = min_distance - radius
        is_collision = collision_distance < 0

        if closest_point is not None:
            collision_normal = position - closest_point
            if np.linalg.norm(collision_normal) > 1e-6:
                collision_normal = collision_normal / np.linalg.norm(collision_normal)

        elapsed_us = (time.perf_counter() - start_time) * 1e6

        return CollisionResult(
            is_collision=is_collision,
            distance=collision_distance,
            closest_point=closest_point,
            collision_normal=collision_normal,
            check_time_us=elapsed_us
        )

    def check_point_cloud_collision(
        self,
        point_cloud: np.ndarray,
        radius: float = 0.1,
        ignore_ids: Optional[List[int]] = None
    ) -> Tuple[bool, List[int], float]:
        """
        Check collision for entire point cloud.

        Args:
            point_cloud: Nx3 array of points
            radius: Agent radius
            ignore_ids: Obstacle IDs to ignore

        Returns:
            Tuple of (has_collision, collision_indices, check_time_us)
        """
        start_time = time.perf_counter()
        ignore_ids = ignore_ids or []

        collision_indices = []
        total_distance = 0.0

        for i, point in enumerate(point_cloud):
            result = self.check_collision(point, radius, ignore_ids)
            total_distance += result.distance

            if result.is_collision:
                collision_indices.append(i)

        has_collision = len(collision_indices) > 0
        avg_distance = total_distance / len(point_cloud) if len(point_cloud) > 0 else 0.0

        elapsed_us = (time.perf_counter() - start_time) * 1e6

        return has_collision, collision_indices, elapsed_us

    def check_trajectory_collision(
        self,
        positions: np.ndarray,
        radius: float = 0.1,
        ignore_ids: Optional[List[int]] = None
    ) -> Tuple[bool, int, List[int], float]:
        """
        Check collision along a trajectory.

        Args:
            positions: Tx3 array of positions
            radius: Agent radius
            ignore_ids: Obstacle IDs to ignore

        Returns:
            Tuple of (has_collision, first_collision_idx, collision_indices, check_time_us)
        """
        start_time = time.perf_counter()
        ignore_ids = ignore_ids or []

        collision_indices = []
        first_collision_idx = -1

        for i, pos in enumerate(positions):
            result = self.check_collision(pos, radius, ignore_ids)

            if result.is_collision:
                collision_indices.append(i)
                if first_collision_idx< 0:
                    first_collision_idx = i

        elapsed_us = (time.perf_counter() - start_time) * 1e6

        return len(collision_indices) > 0, first_collision_idx, collision_indices, elapsed_us

    def _sphere_point_distance(
        self,
        point: np.ndarray,
        center: np.ndarray,
        radius: float
    ) -> Tuple[float, np.ndarray]:
        """Distance from point to sphere surface."""
        diff = point - center
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            closest = center + np.array([0, 0, radius])
        else:
            closest = center + diff / dist * radius
        return dist - radius, closest

    def _box_point_distance(
        self,
        point: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
        rotation: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Distance from point to box surface."""
        # Transform to box frame
        if rotation is not None:
            R_inv = rotation.T
            point_local = R_inv @ (point - center)
        else:
            point_local = point - center

        half_extents = np.array(half_extents)

        # Clamp to box
        closest_local = np.clip(point_local, -half_extents, half_extents)

        # Transform back
        if rotation is not None:
            closest = rotation @ closest_local + center
        else:
            closest = closest_local + center

        dist = np.linalg.norm(point - closest)
        return dist, closest

    def _box_sdf(
        self,
        point: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
        rotation: Optional[np.ndarray] = None
    ) -> float:
        """SDF for box."""
        dist, _ = self._box_point_distance(point, center, half_extents, rotation)
        return dist

    def _cylinder_point_distance(
        self,
        point: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        height: float
    ) -> Tuple[float, np.ndarray]:
        """Distance from point to cylinder surface."""
        # Project point onto axis
        diff = point - center
        proj = np.dot(diff, axis)

        # Clamp to height
        h_clamped = np.clip(proj, 0, height)
        base_point = center + axis * h_clamped

        # Distance in radial direction
        radial = diff - axis * proj
        radial_dist = np.linalg.norm(radial)

        if radial_dist < 1e-6:
            # Point on axis - return closest point on circle
            if abs(proj) < height / 2:
                closest = base_point + axis * radius if proj >= 0 else base_point - axis * radius
            else:
                closest = center + axis * np.clip(proj, 0, height)
        else:
            closest = base_point + radial / radial_dist * radius

        dist = np.linalg.norm(point - closest)

        # Check if we're outside the height range
        if proj < 0:
            dist = np.sqrt(dist**2 + proj**2)
        elif proj > height:
            dist = np.sqrt(dist**2 + (proj - height)**2)

        return dist - radius, closest

    def _cylinder_sdf(
        self,
        point: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        radius: float,
        height: float
    ) -> float:
        """SDF for cylinder."""
        dist, _ = self._cylinder_point_distance(point, center, axis, radius, height)
        return dist

    def _point_cloud_point_distance(
        self,
        point: np.ndarray,
        points: np.ndarray,
        radius: float
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Distance from point to point cloud."""
        if len(points) == 0:
            return float('inf'), None

        # Efficient search using grid
        nearby_ids = self._get_nearby_cells(point, self.max_check_radius)
        candidates = [p for p in points] # Simplified - in production use spatial index
        min_dist = float('inf')
        closest = None

        for p in candidates:
            dist = np.linalg.norm(point - p)
            if dist < min_dist:
                min_dist = dist
                closest = p

        return min_dist - radius, closest

    def _insert_into_grid(self, point: np.ndarray, obstacle_id: int) -> None:
        """Insert obstacle into spatial hash grid."""
        key = self._grid_key(point)
        if key not in self.grid:
            self.grid[key] = []
            self.grid_cells[key] = set()
        self.grid[key].append(obstacle_id)
        self.grid_cells[key].add(obstacle_id)

    def _grid_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Compute spatial hash grid key."""
        res = self.spatial_resolution
        return (
            int(point[0] / res),
            int(point[1] / res),
            int(point[2] / res)
        )

    def _get_nearby_cells(self, point: np.ndarray, radius: float = 0.0) -> set:
        """Get obstacle IDs from nearby grid cells.

        ``radius`` is the query radius (agent radius + a safety margin);
        the search expands over enough grid cells to cover it.
        """
        nearby = set()
        res = self.spatial_resolution
        if res <= 0:
            # Degenerate configuration: fall back to scanning everything.
            for obs in self.obstacles:
                nearby.add(obs["id"])
            return nearby

        span = int(np.ceil(max(radius, res) / res))
        cx, cy, cz = (int(point[0] / res), int(point[1] / res), int(point[2] / res))

        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    ids = self.grid_cells.get(key)
                    if ids:
                        nearby.update(ids)

        return nearby

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()
        self.grid.clear()
        self.grid_cells.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get collision detector statistics."""
        return {
            "num_obstacles": len(self.obstacles),
            "num_grid_cells": len(self.grid),
            "obstacle_types": {
                obs["type"]: sum(1 for o in self.obstacles if o["type"] == obs["type"])
                for obs in self.obstacles
            } if self.obstacles else {}
        }


def create_robot_collision_checker(
    link_radii: List[float],
    link_lengths: List[float]
) -> Callable:
    """
    Create a robot-specific collision checker.

    Args:
        link_radii: List of radii for each link
        link_lengths: List of lengths for each link

    Returns:
        Collision checking function
    """
    def check_robot_collision(
        positions: List[np.ndarray],
        detector: CollisionDetector,
        ignore_ids: Optional[List[int]] = None
    ) -> Tuple[bool, int, float]:
        """
        Check collision for robot links.

        Args:
            positions: List of link positions
            detector: CollisionDetector instance
            ignore_ids: Obstacles to ignore

        Returns:
            Tuple of (collision, link_idx, distance)
        """
        ignore_ids = ignore_ids or []

        for link_idx, pos in enumerate(positions):
            radius = link_radii[link_idx] if link_idx < len(link_radii) else 0.1
            result = detector.check_collision(pos, radius, ignore_ids)

            if result.is_collision:
                return True, link_idx, result.distance

        return False, -1, 0.0

    return check_robot_collision