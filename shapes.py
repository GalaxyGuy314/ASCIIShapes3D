import math
import numpy as np
import trimesh
from numba import njit


@njit(fastmath=True)
def _compute_projection(
        mesh_positions,
        current_angle,
        orbit_angles,
        fov,
        camera_distance,
        resolution
):
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

    object_rotation = (
            rotation_matrix_x
            @ rotation_matrix_y
            @ rotation_matrix_z
    )

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

    total_matrix = (
            orbit_matrix_y
            @ orbit_matrix_p
            @ object_rotation
    )

    fov_rad = math.radians(fov)
    focal_length = (
            (resolution[1] / 2.0)
            / math.tan(fov_rad / 2.0)
    )

    n_vertices = mesh_positions.shape[0]
    world_positions = np.zeros((n_vertices, 4), dtype=np.float64)
    projected_positions = np.zeros((n_vertices, 2), dtype=np.float64)

    #For every vertex, compute the world position and projection
    for i in range(n_vertices):
        wp = mesh_positions[i] @ total_matrix
        world_positions[i] = wp

        final_x = wp[0]
        final_y = wp[1]
        final_z = wp[2] + camera_distance

        if final_z <= 0.001:
            final_z = 0.001

        projection_factor = focal_length / final_z

        screen_x = (
                resolution[0] / 2.0
                + final_x * projection_factor * 1.666
        )

        screen_y = (
                resolution[1] / 2.0
                + final_y * projection_factor
        )

        projected_positions[i, 0] = screen_x
        projected_positions[i, 1] = screen_y

    return world_positions, projected_positions


def _check_lighting(vertex, shape, light_position):
    """
    Cast a ray from the light source to this vertex and check if
    any face of the shape blocks the path (shadow test).
    """
    # vertex is either class Vertex or SubVertex
    light_pos = np.array(light_position, dtype=np.float64)
    ray_dir = vertex.world_position[:3] - light_pos
    ray_length = np.linalg.norm(ray_dir)

    if ray_length < 1e-6:
        vertex.is_lit = True
        return True

    ray_dir /= ray_length
    epsilon = 1e-4

    # Build a set of raw vertex indices that belong to this vertex (or parents if SubVertex)
    excluded_indices = set()
    if hasattr(vertex, 'parent_vertices') and vertex.parent_vertices:
        for p in vertex.parent_vertices:
            for idx, rv in enumerate(shape.vertices):
                if rv == p or (hasattr(p, 'mesh_position') and np.allclose(rv.mesh_position, p.mesh_position)):
                    excluded_indices.add(idx)
    else:
        for idx, rv in enumerate(shape.vertices):
            if rv == vertex or np.allclose(rv.mesh_position, vertex.mesh_position):
                excluded_indices.add(idx)

    for face_idx, face in enumerate(shape.faces):
        # Skip faces that are part of this vertex to prevent self-intersection
        if any(vi in excluded_indices for vi in face.vertices_indices):
            continue

        v0 = shape.vertices[face.vertices_indices[0]].world_position[:3]
        v1 = shape.vertices[face.vertices_indices[1]].world_position[:3]
        v2 = shape.vertices[face.vertices_indices[2]].world_position[:3]

        # Möller–Trumbore ray-triangle intersection algorithm
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            continue

        f = 1.0 / a
        s = light_pos - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            continue

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)

        if v < 0.0 or u + v > 1.0:
            continue

        t = f * np.dot(edge2, q)

        # Check if intersection is between the light source and the vertex
        if epsilon < t < ray_length - epsilon:
            vertex.is_lit = False
            return False

    vertex.is_lit = True
    return True



class Vertex:
    def __init__(self, mesh_position):
        self.mesh_position = np.array([mesh_position[0], mesh_position[1], mesh_position[2], 1.0], dtype=np.float64)
        self.is_lit = None
        self.world_position = np.zeros(4)
        self.projected_position = np.zeros(2)

    def update_lighting(self, shape, light_position):
        _check_lighting(self, shape, light_position)


#for now skip subvertices and get everything working
class SubVertex:
    def __init__(self, parent_vertices):
        self.parent_vertices = sorted(parent_vertices)
        self.world_position = np.zeros(3, dtype=np.float64)
        self.is_lit = None

    def update_world_position(self):
        self.world_position = np.mean(
            [v.world_position for v in self.parent_vertices],
            axis=0
        )

    def update_lighting(self, shape, light_position):
        _check_lighting(self, shape, light_position)



class Edge:
    def __init__(self, vertices_indices):
        self.vertices_indices = vertices_indices

    def length(self, vertices_positions):
        p1 = vertices_positions[self.vertices_indices[0]]
        p2 = vertices_positions[self.vertices_indices[1]]
        return np.linalg.norm(p1 - p2)


class Face:
    def __init__(self, vertices_indices):
        self.vertices_indices = vertices_indices
        self.normal = None
        self.light_level = None

    #the normals should be updated directly after rotation.
    def _update_normal(self, base_vertices):
        v0 = base_vertices[self.vertices_indices[0]].world_position[:3]
        v1 = base_vertices[self.vertices_indices[1]].world_position[:3]
        v2 = base_vertices[self.vertices_indices[2]].world_position[:3]

        edge_vector1 = v1 - v0
        edge_vector2 = v2 - v0

        normal = np.cross(
            edge_vector1,
            edge_vector2
        )

        length = np.linalg.norm(normal)

        if length > 0.0:
            self.normal = normal / length
        else:
            self.normal = np.zeros(3, dtype=np.float64) #

    def _update_light_level(
            self,
            base_vertices,
            shape,
            light_position,
            base_light_level=0.1
    ):
        vertices = [
            base_vertices[i]
            for i in self.vertices_indices
        ]

        n_samples = len(vertices)

        light_level = base_light_level
        remaining_light_level = 1.0 - base_light_level

        for vertex in vertices:
            light_direction = (
                    light_position - vertex.world_position[:3]
            )

            distance = np.linalg.norm(light_direction)

            if distance < 1e-8:
                continue

            light_direction /= distance

            vertex.update_lighting(
                shape,
                light_position
            )

            if vertex.is_lit:
                diffuse = max(
                    0.0,
                    np.dot(self.normal, light_direction)
                )

                light_level += (
                        diffuse
                        * remaining_light_level
                        / n_samples
                )

        self.light_level = np.clip(
            light_level,
            0.0,
            1.0
        )

    def update(self, base_vertices, shape, light_position):
        self._update_normal(base_vertices)
        self._update_light_level(base_vertices, shape, light_position)


class Shape:
    def __init__(
            self,
            filename,
            resolution,
            rotation_speed_vector,
            fov,
            camera_distance
    ):
        self.filename = filename

        self.current_angle = [0.0, 0.0, 0.0]

        self.vertices, self.faces, self.edges = (
            self._load_mesh(filename)
        )



    def _load_mesh(self, filename):
        try:
            mesh = trimesh.load(
                filename,
                force="mesh",
                process=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load mesh '{filename}': {exc}"
            ) from exc

        if mesh is None or mesh.is_empty:
            raise ValueError(
                f"Mesh '{filename}' contains no geometry."
            )

        raw_vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64
        )

        raw_faces = np.asarray(
            mesh.faces,
            dtype=np.int64
        )

        if len(raw_vertices) == 0 or len(raw_faces) == 0:
            raise ValueError(
                f"Mesh '{filename}' contains no usable geometry."
            )

        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError(
                f"'{filename}' contains non-triangular faces."
            )

        minimum = raw_vertices.min(axis=0)
        maximum = raw_vertices.max(axis=0)
        centre = (minimum + maximum) / 2.0
        raw_vertices -= centre

        minimum = raw_vertices.min(axis=0)
        maximum = raw_vertices.max(axis=0)
        dimensions = maximum - minimum
        largest_dimension = np.max(dimensions)

        if largest_dimension <= 0.0:
            raise ValueError(
                f"Mesh '{filename}' has zero size."
            )

        raw_vertices /= largest_dimension

        vertices = [Vertex(pos) for pos in raw_vertices]

        faces = [Face(list(f)) for f in raw_faces]

        edges_set = set()
        for face in raw_faces:
            a, b, c = face
            edges_set.add(tuple(sorted((int(a), int(b)))))
            edges_set.add(tuple(sorted((int(b), int(c)))))
            edges_set.add(tuple(sorted((int(c), int(a)))))

        edges = [Edge(list(e)) for e in edges_set]

        return vertices, faces, edges

    def update(self,
               orbit_angles,
               light_position,
               fov,
               resolution,
               camera_distance
               ):
        # first rotate the shape.
        # then update all faces.

        mesh_positions = np.array([v.mesh_position for v in self.vertices], dtype=np.float64)
        world_pos, proj_pos = _compute_projection(
            mesh_positions,
            self.current_angle,
            orbit_angles,
            fov,
            camera_distance,
            resolution
        )

        for i, vertex in enumerate(self.vertices):
            vertex.world_position = world_pos[i]
            vertex.projected_position = proj_pos[i]

        for face in self.faces:
            face.update(self.vertices,
                        self,
                        light_position
                        )