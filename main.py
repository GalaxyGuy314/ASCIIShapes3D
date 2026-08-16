import math
import tkinter as tk
from controls import ControlPanel
from shapes import Cube, Pyramid

RESOLUTION = [int(128 * 1.666), 128]
CAMERA_DISTANCE = 2.5
FOV = 60.0
PI = math.pi
ROTATION_SPEED = [0.0, PI / 2, PI / 6]
DT = 0.03

shape = "cube"


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
      self.orbit_angles[0] -= rot_step  # Yaw
    elif key == "d":
      self.orbit_angles[0] += rot_step  # Yaw
    elif event.keysym == "Up":
      self.camera_distance = max(1.0, self.camera_distance - dist_step)  # Zoom in
      self.control_panel.dist_slider.set(self.camera_distance)
    elif event.keysym == "Down":
      self.camera_distance = min(15.0, self.camera_distance + dist_step)  # Zoom out
      self.control_panel.dist_slider.set(self.camera_distance)

  def draw_line(self, x0, y0, x1, y1, buffer, char="#"):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
      if 0 <= x0 < self.resolution[0] and 0 <= y0 < self.resolution[1]:
        buffer[y0][x0] = char
      if x0 == x1 and y0 == y1:
        break
      e2 = 2 * err
      if e2 > -dy:
        err -= dy
        x0 += sx
      if e2 < dx:
        err += dx
        y0 += sy

  def animate(self):
    buffer = [
        [" " for _ in range(self.resolution[0])]
        for _ in range(self.resolution[1])
    ]

    self.shape.fov = self.fov
    self.shape.camera_distance = self.camera_distance
    self.shape.rotation_speed_vector = self.rotation_speed_vector
    self.shape.orbit_angles = self.orbit_angles

    try:
      if float(self.control_panel.dist_slider.get()) != self.camera_distance:
        self.control_panel.dist_slider.set(self.camera_distance)
    except Exception:
      pass

    projected_vertices = self.shape.rotate_and_project()

    height = self.resolution[1]
    projected_vertices = [
        [p[0], height - p[1], *p[2:]] for p in projected_vertices
    ]

    for edge in self.shape.edges:
      p1 = projected_vertices[edge[0]]
      p2 = projected_vertices[edge[1]]
      self.draw_line(p1[0], p1[1], p2[0], p2[1], buffer, char=".")

    for p in projected_vertices:
      if 0 <= p[0] < self.resolution[0] and 0 <= p[1] < self.resolution[1]:
        buffer[p[1]][p[0]] = "O"

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