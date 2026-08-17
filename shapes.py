import math
import numpy as np
from numba import njit

@njit(fastmath=True)
def _compute_projection(base_vertices, current_angle, orbit_angles, fov, camera_distance, resolution):
    ax, ay, az = current_angle
    sin_x, cos_x = math.sin(ax), math.cos(ax)
    sin_y, cos_y = math.sin(ay), math.cos(ay)
    sin_z, cos_z = math.sin(az), math.cos(az)

    orbit_yaw, orbit_pitch = orbit_angles
    sin_orbit_yaw, cos_orbit_yaw = math.sin(orbit_yaw), math.cos(orbit_yaw)
    sin_orbit_pitch, cos_orbit_pitch = math.sin(orbit_pitch), math.cos(orbit_pitch)

    rotation_matrix_x = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x, 0.0],
        [0.0, sin_x, cos_x, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    rotation_matrix_y = np.array([
        [cos_y, 0.0, sin_y, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    rotation_matrix_z = np.array([
        [cos_z, -sin_z, 0.0, 0.0],
        [sin_z, cos_z, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    object_rotation = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    orbit_matrix_y = np.array([
        [cos_orbit_yaw, 0.0, sin_orbit_yaw, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_orbit_yaw, 0.0, cos_orbit_yaw, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    orbit_matrix_p = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_orbit_pitch, -sin_orbit_pitch, 0.0],
        [0.0, sin_orbit_pitch, cos_orbit_pitch, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    total_matrix = (orbit_matrix_y @ orbit_matrix_p) @ object_rotation

    fov_rad = math.radians(fov)
    focal_length = (resolution[1] / 2) / math.tan(fov_rad / 2)

    n_vertices = len(base_vertices)
    projected_vertices = np.zeros((n_vertices, 2), dtype=np.float64)
    transformed_3d = np.zeros((n_vertices, 3), dtype=np.float64)

    for i in range(n_vertices):
        transformed = base_vertices[i] @ total_matrix

        final_x = transformed[0]
        final_y = transformed[1]
        final_z = transformed[2] + camera_distance

        if final_z <= 0.001:
            final_z = 0.001

        transformed_3d[i] = [final_x, final_y, final_z]

        projection_factor = focal_length / final_z

        screen_x = resolution[0] / 2 + final_x * projection_factor * 1.666
        screen_y = resolution[1] / 2 + final_y * projection_factor

        projected_vertices[i, 0] = screen_x
        projected_vertices[i, 1] = screen_y

    return projected_vertices, transformed_3d


@njit(fastmath=True)
def _compute_normals(transformed_3d, faces):
    n_faces = len(faces)
    surface_normals = np.zeros((n_faces, 3), dtype=np.float64)

    for i in range(n_faces):
        f = faces[i]
        v0 = transformed_3d[f[0]]
        v1 = transformed_3d[f[1]]
        v2 = transformed_3d[f[2]]

        edge_vector1 = v1 - v0
        edge_vector2 = v2 - v0

        normal = np.cross(edge_vector1, edge_vector2)
        length = np.linalg.norm(normal)

        if length > 0.0:
            surface_normals[i] = normal / length
        else:
            surface_normals[i] = 0.0

    return surface_normals


class Shape:
    """Base class for handling 3D projection, text rendering, and orbiting."""

    def __init__(
            self,
            resolution,
            base_vertices,
            edges,
            faces,
            rotation_speed,
            fov,
            camera_dist,
    ):
        self.resolution = resolution
        self.base_vertices = np.array(base_vertices, dtype=np.float64)
        self.edges = edges
        self.faces = np.array(faces, dtype=np.int64) if faces is not None else np.zeros((0, 3), dtype=np.int64)
        self.rotation_speed_vector = rotation_speed
        self.fov = fov
        self.camera_distance = camera_dist
        self.current_angle = [0.0, 0.0, 0.0]
        self.orbit_angles = [0.0, 0.0]

    def rotate_and_project(self):
        projected_vertices, transformed_3d = _compute_projection(
            self.base_vertices,
            self.current_angle,
            self.orbit_angles,
            self.fov,
            self.camera_distance,
            self.resolution
        )
        surface_normals = _compute_normals(transformed_3d, self.faces)
        return projected_vertices, surface_normals, transformed_3d


class Cube(Shape):
    """Subclass representing a 3D Cube with preset vertices and edges."""

    def __init__(self, resolution, rotation_speed, fov, camera_dist):
        cube_vertices = [
            [-0.5, -0.5, -0.5, 1.0],  # 0
            [-0.5, -0.5, 0.5, 1.0],  # 1
            [0.5, -0.5, -0.5, 1.0],  # 2
            [0.5, -0.5, 0.5, 1.0],  # 3
            [-0.5, 0.5, -0.5, 1.0],  # 4
            [-0.5, 0.5, 0.5, 1.0],  # 5
            [0.5, 0.5, -0.5, 1.0],  # 6
            [0.5, 0.5, 0.5, 1.0],  # 7
        ]

        cube_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        cube_faces = [
            (0, 2, 3), (0, 3, 1),

            (4, 5, 7), (4, 7, 6),

            (1, 3, 7), (1, 7, 5),

            (0, 4, 6), (0, 6, 2),

            (0, 1, 5), (0, 5, 4),

            (2, 6, 7), (2, 7, 3),
        ]

        super().__init__(
            resolution, cube_vertices, cube_edges, cube_faces, rotation_speed, fov, camera_dist
        )


class Pyramid(Shape):
    """Subclass representing a 3D Square Pyramid with preset vertices and edges."""

    def __init__(self, resolution, rotation_speed, fov, camera_dist):
        pyramid_vertices = [
            [-0.5, -0.5, -0.5, 1.0],
            [-0.5, -0.5, 0.5, 1.0],
            [0.5, -0.5, -0.5, 1.0],
            [0.5, -0.5, 0.5, 1.0],
            [0.0, 0.5, 0.0, 1.0],
        ]

        pyramid_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (0, 4), (1, 4), (2, 4), (3, 4),
        ]

        pyramid_faces = [
            (0, 1, 4),
            (1, 3, 4),
            (3, 2, 4),
            (2, 0, 4),

            (0, 3, 1),
            (0, 2, 3)
        ]

        super().__init__(
            resolution, pyramid_vertices, pyramid_edges, pyramid_faces, rotation_speed, fov, camera_dist
        )