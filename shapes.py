import math


class Shape:
  """Base class for handling 3D projection, text rendering, and orbiting."""

  def __init__(
      self,
      resolution,
      base_vertices,
      edges,
      rotation_speed,
      fov,
      camera_dist,
  ):
    self.resolution = resolution
    self.base_vertices = base_vertices
    self.edges = edges
    self.rotation_speed_vector = rotation_speed
    self.fov = fov
    self.camera_distance = camera_dist
    self.current_angle = [0.0, 0.0, 0.0]
    self.orbit_angles = [0.0, 0.0]

  def rotate_and_project(self):
    ax, ay, az = self.current_angle
    sin_x, cos_x = math.sin(ax), math.cos(ax)
    sin_y, cos_y = math.sin(ay), math.cos(ay)
    sin_z, cos_z = math.sin(az), math.cos(az)

    oy, op = self.orbit_angles
    sin_oy, cos_oy = math.sin(oy), math.cos(oy)
    sin_op, cos_op = math.sin(op), math.cos(op)

    projected_vertices = []

    fov_rad = math.radians(self.fov)
    focal_length = (self.resolution[1] / 2) / math.tan(fov_rad / 2)

    for vertex in self.base_vertices:
      vx, vy, vz, vw = vertex

      rx = (
          vx * (cos_y * cos_z + sin_y * sin_x * sin_z)
          + vy * (cos_x * sin_z)
          + vz * (-sin_y * cos_z + cos_y * sin_x * sin_z)
      )
      ry = (
          vx * (-cos_y * sin_z + sin_y * sin_x * cos_z)
          + vy * (cos_x * cos_z)
          + vz * (sin_y * sin_z + cos_y * sin_x * cos_z)
      )
      rz = vx * (sin_y * cos_x) + vy * (-sin_x) + vz * (cos_y * cos_x)

      ox = rx * cos_oy + rz * sin_oy
      oy_val = ry
      oz = -rx * sin_oy + rz * cos_oy

      final_x = ox
      final_y = oy_val * cos_op - oz * sin_op
      final_z = oy_val * sin_op + oz * cos_op

      final_z += self.camera_distance

      if final_z <= 0.001:
        final_z = 0.001

      projection_factor = focal_length / final_z

      screen_x = int(self.resolution[0] / 2 + final_x * projection_factor * 1.666)
      screen_y = int(self.resolution[1] / 2 + final_y * projection_factor)

      projected_vertices.append((screen_x, screen_y))

    return projected_vertices


class Cube(Shape):
  """Subclass representing a 3D Cube with preset vertices and edges."""

  def __init__(self, resolution, rotation_speed, fov, camera_dist):
    cube_vertices = [
        [-0.5, -0.5, -0.5, 1.0],
        [-0.5, -0.5, 0.5, 1.0],
        [0.5, -0.5, -0.5, 1.0],
        [0.5, -0.5, 0.5, 1.0],
        [-0.5, 0.5, -0.5, 1.0],
        [-0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, -0.5, 1.0],
        [0.5, 0.5, 0.5, 1.0],
    ]

    cube_edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    super().__init__(
        resolution, cube_vertices, cube_edges, rotation_speed, fov, camera_dist
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
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ]

    super().__init__(
        resolution, pyramid_vertices, pyramid_edges, rotation_speed, fov, camera_dist
    )