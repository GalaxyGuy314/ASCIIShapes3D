import math
import tkinter as tk
import numpy as np
from controls import ControlPanel
from shapes import Cube, Pyramid
import numba

RESOLUTION = [int(128 * 1.666), 128]
CAMERA_DISTANCE = 1.9
FOV = 60.0
PI = np.pi
ROTATION_SPEED = [0.0, 0.0, 0.0]
DT = 0.03

shape = "pyramid"


class App:
    """Main application controller managing the view, loop, and input."""

    def __init__(self, root):
        self.root = root
        self.root.title("Text-Based 3D Visualizer")

        self.resolution = RESOLUTION
        self.fov = FOV
        self.camera_distance = CAMERA_DISTANCE
        self.rotation_speed_vector = ROTATION_SPEED
        self.dt = DT
        self.orbit_angles = [0.0, 0.0]

        self.light_source = [[0.0, 0.0, -3.5], [0.0, 0.0, 0.0]]
        self.display_edges = False
        self.display_faces = True

        self.gradient = """.'",;:!~+*#@$"""

        if shape == "cube":
            self.shape = Cube(
                self.resolution, self.rotation_speed_vector, self.fov, self.camera_distance
            )
        elif shape == "pyramid":
            self.shape = Pyramid(
                self.resolution, self.rotation_speed_vector, self.fov, self.camera_distance
            )

        self.text_widget = tk.Text(
            root,
            width=self.resolution[0],
            height=self.resolution[1],
            bg="black",
            fg="white",
            font=("Courier", 6),
            insertbackground="white",
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        self.text_widget.pack(fill=tk.BOTH, expand=False)

        self.root.bind_all("<KeyPress>", self.handle_keypress)
        self.control_panel = ControlPanel(root, self)
        self.animate()

    def handle_keypress(self, event):
        rot_step = 0.1
        dist_step = 0.2
        key = event.keysym.lower()

        if key == "w":
            self.orbit_angles[1] += rot_step  # Pitch
        elif key == "s":
            self.orbit_angles[1] -= rot_step  # Pitch
        elif key == "a":
            self.orbit_angles[0] += rot_step  # Yaw
        elif key == "d":
            self.orbit_angles[0] -= rot_step  # Yaw
        elif event.keysym == "Up":
            self.camera_distance = max(1.0, self.camera_distance - dist_step)  # Zoom in
            self.control_panel.dist_slider.set(self.camera_distance)
        elif event.keysym == "Down":
            self.camera_distance = min(15.0, self.camera_distance + dist_step)  # Zoom out
            self.control_panel.dist_slider.set(self.camera_distance)

    def normalise_vector(self, p1, p2):
        diff = p1 - p2
        length = np.linalg.norm(diff)
        if length > 0.0:
            return diff / length
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def draw_line(self, x0, y0, x1, y1, buffer, z_buffer, char="#", depth=0.0):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        width = self.resolution[0]
        height = self.resolution[1]

        while True:
            if 0 <= x0 < width and 0 <= y0 < height:
                if depth < z_buffer[y0, x0]:
                    z_buffer[y0, x0] = depth
                    buffer[y0, x0] = char
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_surface(self, projected_vertices, face, surface_normals, index, buffer, z_buffer, light_direction,
                     transformed_3d):
        v_indices = [face[0], face[1], face[2]]
        points = sorted(
            [(projected_vertices[i], transformed_3d[i]) for i in v_indices],
            key=lambda item: item[0][1]
        )

        (p1, z1), (p2, z2), (p3, z3) = points

        surface_normal = surface_normals[index]

        intensity = max(0.0, np.dot(surface_normal, light_direction))
        char_index = int(intensity * (len(self.gradient) - 1))
        shaded_char = self.gradient[char_index]

        width = self.resolution[0]
        height = self.resolution[1]

        def draw_span(y, x_start, x_end, z_start, z_end):
            if 0 <= y < height:
                x_s_int = int(round(min(x_start, x_end)))
                x_e_int = int(round(max(x_start, x_end)))
                span_len = x_e_int - x_s_int

                for x in range(max(0, x_s_int), min(width, x_e_int + 1)):
                    t = 0.0 if span_len == 0 else (x - x_start) / (x_end - x_start + 1e-12)
                    current_z = z_start + (z_end - z_start) * t

                    if current_z < z_buffer[y, x]:
                        z_buffer[y, x] = current_z
                        buffer[y, x] = shaded_char

        y1, y2, y3 = int(p1[1]), int(p2[1]), int(p3[1])

        if y3 == y1:
            return

        total_height = y3 - y1

        for i in range(total_height + 1):
            current_y = y1 + i
            if not (0 <= current_y < height):
                continue

            second_half = i > y2 - y1 or y2 == y1
            segment_height = y3 - y2 if second_half else y2 - y1
            if segment_height == 0:
                continue

            alpha = i / total_height
            beta = (i - (y2 - y1)) / segment_height if second_half else i / (y2 - y1 + 1e-12)

            if second_half:
                ax = p1[0] + (p3[0] - p1[0]) * alpha
                az = z1[2] + (z3[2] - z1[2]) * alpha
                bx = p2[0] + (p3[0] - p2[0]) * beta
                bz = z2[2] + (z3[2] - z2[2]) * beta
            else:
                ax = p1[0] + (p3[0] - p1[0]) * alpha
                az = z1[2] + (z3[2] - z1[2]) * alpha
                bx = p1[0] + (p2[0] - p1[0]) * beta
                bz = z1[2] + (z2[2] - z1[2]) * beta

            draw_span(current_y, ax, bx, az, bz)

    def draw_edges(self, projected_vertices, buffer, z_buffer, transformed_3d):
        for edge in self.shape.edges:
            p1 = projected_vertices[edge[0]]
            p2 = projected_vertices[edge[1]]
            avg_z = (transformed_3d[edge[0]][2] + transformed_3d[edge[1]][2]) / 2.0
            self.draw_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), buffer, z_buffer, char=".", depth=avg_z)

    def draw_vertices(self, projected_vertices, buffer, z_buffer, transformed_3d):
        width = self.resolution[0]
        height = self.resolution[1]
        for i, p in enumerate(projected_vertices):
            px, py = int(p[0]), int(p[1])
            if 0 <= px < width and 0 <= py < height:
                depth = transformed_3d[i][2]
                if depth < z_buffer[py, px]:
                    z_buffer[py, px] = depth
                    buffer[py, px] = "O"

    def draw_faces(self, projected_vertices, buffer, z_buffer, surface_normals, transformed_3d, light_direction):
        face_depths = []
        for index, face in enumerate(self.shape.faces):
            z1 = transformed_3d[face[0]][2]
            z2 = transformed_3d[face[1]][2]
            z3 = transformed_3d[face[2]][2]

            avg_z = (z1 + z2 + z3) / 3.0
            face_depths.append((avg_z, index, face))

        face_depths.sort(key=lambda item: item[0], reverse=True)

        for avg_z, index, face in face_depths:
            self.draw_surface(projected_vertices, face, surface_normals, index, buffer, z_buffer, light_direction,
                              transformed_3d)

    def animate(self):
        buffer = np.full((self.resolution[1], self.resolution[0]), " ", dtype="<U1")
        z_buffer = np.full((self.resolution[1], self.resolution[0]), np.inf, dtype=np.float64)

        self.shape.fov = self.fov
        self.shape.camera_distance = self.camera_distance
        self.shape.rotation_speed_vector = self.rotation_speed_vector
        self.shape.orbit_angles = self.orbit_angles

        try:
            if float(self.control_panel.dist_slider.get()) != self.camera_distance:
                self.control_panel.dist_slider.set(self.camera_distance)
        except Exception:
            pass

        projected_vertices, surface_normals, transformed_3d = self.shape.rotate_and_project()

        height = self.resolution[1]
        projected_vertices_screen = np.zeros_like(projected_vertices, dtype=np.float64)
        for i in range(len(projected_vertices)):
            projected_vertices_screen[i, 0] = projected_vertices[i, 0]
            projected_vertices_screen[i, 1] = height - projected_vertices[i, 1]

        light_p1 = np.array(self.light_source[0], dtype=np.float64)
        light_p2 = np.array(self.light_source[1], dtype=np.float64)
        light_direction = self.normalise_vector(light_p1, light_p2)

        if self.display_faces:
            self.draw_faces(projected_vertices_screen, buffer, z_buffer, surface_normals, transformed_3d,
                            light_direction)
        elif self.display_edges:
            self.draw_edges(projected_vertices_screen, buffer, z_buffer, transformed_3d)

        self.draw_vertices(projected_vertices_screen, buffer, z_buffer, transformed_3d)

        output = "\n".join("".join(row) for row in buffer)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, output)

        self.shape.current_angle = [
            a + (self.dt * s)
            for a, s in zip(self.shape.current_angle, self.shape.rotation_speed_vector)
        ]

        self.root.after(int(1000 * DT), self.animate)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()