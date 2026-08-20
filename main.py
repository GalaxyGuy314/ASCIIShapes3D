import math
import tkinter as tk
import numpy as np

from controls import ControlPanel
from shapes import Shape, SubVertex

RESOLUTION = [int(64 * 1.666), 64]
CAMERA_DISTANCE = 1.9
FOV = 60.0
ROTATION_SPEED = [0.0, np.pi / 6, 0.0]
DT = 0.03
MODEL = "models/cube.obj"


class App:
    """Main application controller."""

    def __init__(self, root):
        self.root = root
        self.root.title("Text-Based 3D Visualizer")

        self.resolution = RESOLUTION
        self.fov = FOV
        self.camera_distance = CAMERA_DISTANCE

        self.rotation_speed_vector = ROTATION_SPEED.copy()
        self.dt = DT

        self.orbit_angles = [0.0, 0.0]
        self.light_position = [-2.0, 0.0, -2.5]

        self.display_edges = False
        self.display_faces = True
        self.display_vertices = True

        self.gradient = "-~+*I#@$"

        self.shape = Shape(
            MODEL,
            self.resolution,
            self.rotation_speed_vector,
            self.fov,
            self.camera_distance
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

        self.text_widget.pack(
            fill=tk.BOTH,
            expand=False
        )

        self.root.bind_all(
            "<KeyPress>",
            self.handle_keypress
        )

        self.control_panel = ControlPanel(
            root,
            self
        )

        self.animate()

    def handle_keypress(self, event):
        rot_step = 0.1
        dist_step = 0.2
        key = event.keysym.lower()

        if key == "w":
            self.orbit_angles[1] += rot_step
        elif key == "s":
            self.orbit_angles[1] -= rot_step
        elif key == "a":
            self.orbit_angles[0] += rot_step
        elif key == "d":
            self.orbit_angles[0] -= rot_step
        elif event.keysym == "Up":
            self.camera_distance = max(1.0, self.camera_distance - dist_step)
            self.control_panel.dist_slider.set(self.camera_distance)
        elif event.keysym == "Down":
            self.camera_distance = min(15.0, self.camera_distance + dist_step)
            self.control_panel.dist_slider.set(self.camera_distance)

    def normalise_vector(self, p1, p2):
        difference = p1 - p2
        length = np.linalg.norm(difference)
        if length > 0.0:
            return difference / length
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def draw_line(
        self,
        x0,
        y0,
        x1,
        y1,
        buffer,
        z_buffer,
        char=".",
        depth=0.0
    ):
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



    def draw_surface(
        self,
        projected_vertices,
        face,
        buffer,
        z_buffer,
        transformed_3d,
        transformed_vertices
    ):
        v_indices = face.vertices_indices

        p1 = projected_vertices[v_indices[0]]
        p2 = projected_vertices[v_indices[1]]
        p3 = projected_vertices[v_indices[2]]

        z1 = transformed_3d[v_indices[0]][2]
        z2 = transformed_3d[v_indices[1]][2]
        z3 = transformed_3d[v_indices[2]][2]

        intensity = max(
            0.0,
            face.light_level if face.light_level is not None else 0.0
        )

        char_index = int(
            intensity * (len(self.gradient) - 1)
        )
        shaded_char = self.gradient[char_index]

        width = self.resolution[0]
        height = self.resolution[1]

        min_x = max(0, int(math.floor(min(p1[0], p2[0], p3[0]))))
        max_x = min(width - 1, int(math.ceil(max(p1[0], p2[0], p3[0]))))
        min_y = max(0, int(math.floor(min(p1[1], p2[1], p3[1]))))
        max_y = min(height - 1, int(math.ceil(max(p1[1], p2[1], p3[1]))))

        def edge_fn(a, b, c):
            return (
                (c[0] - a[0]) * (b[1] - a[1])
                - (c[1] - a[1]) * (b[0] - a[0])
            )

        area = edge_fn(p1, p2, p3)
        if abs(area) < 1e-5:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                w1 = edge_fn(p2, p3, p)
                w2 = edge_fn(p3, p1, p)
                w3 = edge_fn(p1, p2, p)

                if (
                    (w1 >= 0 and w2 >= 0 and w3 >= 0)
                    or
                    (w1 <= 0 and w2 <= 0 and w3 <= 0)
                ):
                    w1 /= area
                    w2 /= area
                    w3 /= area

                    current_z = (
                        w1 * z1
                        + w2 * z2
                        + w3 * z3
                    )

                    if current_z < z_buffer[y, x]:
                        z_buffer[y, x] = current_z
                        buffer[y, x] = shaded_char

    def draw_edges(
        self,
        projected_vertices,
        buffer,
        z_buffer,
        transformed_3d
    ):
        for edge in self.shape.edges:
            p1 = projected_vertices[edge.vertices_indices[0]]
            p2 = projected_vertices[edge.vertices_indices[1]]

            avg_z = (
                transformed_3d[edge.vertices_indices[0]][2]
                + transformed_3d[edge.vertices_indices[1]][2]
            ) / 2.0

            self.draw_line(
                int(p1[0]),
                int(p1[1]),
                int(p2[0]),
                int(p2[1]),
                buffer,
                z_buffer,
                char=".",
                depth=avg_z
            )

    def draw_vertices(
        self,
        projected_vertices,
        buffer,
        z_buffer,
        transformed_3d
    ):
        width = self.resolution[0]
        height = self.resolution[1]

        for i, p in enumerate(projected_vertices):
            px = int(p[0])
            py = int(p[1])

            if 0 <= px < width and 0 <= py < height:
                depth = transformed_3d[i][2]
                if depth < z_buffer[py, px]:
                    z_buffer[py, px] = depth
                    buffer[py, px] = "O"

    def draw_faces(
        self,
        projected_vertices,
        buffer,
        z_buffer,
        transformed_3d
    ):
        face_depths = []

        for index, face in enumerate(self.shape.faces):
            z1 = self.shape.vertices[face.vertices_indices[0]].world_position[2]
            z2 = transformed_3d[face.vertices_indices[1]][2]
            z3 = transformed_3d[face.vertices_indices[2]][2]

            avg_z = (z1 + z2 + z3) / 3.0
            face_depths.append((avg_z, index, face))

        face_depths.sort(key=lambda item: item[0], reverse=True)

        for avg_z, index, face in face_depths:
            self.draw_surface(
                projected_vertices,
                face,
                buffer,
                z_buffer,
                transformed_3d,
                self.shape.vertices
            )

    def animate(self):
        buffer = np.full(
            (self.resolution[1], self.resolution[0]),
            " ",
            dtype="<U1"
        )
        z_buffer = np.full(
            (self.resolution[1], self.resolution[0]),
            np.inf,
            dtype=np.float64
        )


        try:
            if float(self.control_panel.dist_slider.get()) != self.camera_distance:
                self.control_panel.dist_slider.set(self.camera_distance)
        except Exception:
            pass

        # update the shape with the current values
        self.shape.update(self.orbit_angles,
                          self.light_position,
                          self.fov,
                          self.resolution,
                          self.camera_distance
                          )

        height = self.resolution[1]
        projected_vertices_screen = np.zeros((len(self.shape.vertices), 2), dtype=np.float64)
        transformed_3d = np.zeros((len(self.shape.vertices), 4), dtype=np.float64)

        for i in range(len(self.shape.vertices)):
            projected_vertices_screen[i, 0] = self.shape.vertices[i].projected_position[0]
            projected_vertices_screen[i, 1] = height - self.shape.vertices[i].projected_position[1]
            transformed_3d[i] = self.shape.vertices[i].world_position

        #keep it to faces for now
        if self.display_faces:
            self.draw_faces(
                projected_vertices_screen,
                buffer,
                z_buffer,
                transformed_3d
            )

        output = "\n".join("".join(row) for row in buffer)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, output)

        self.shape.current_angle = [
            angle + self.dt * speed
            for angle, speed in zip(
                self.shape.current_angle,
                self.rotation_speed_vector
            )
        ]

        self.root.after(int(1000 * self.dt), self.animate)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()