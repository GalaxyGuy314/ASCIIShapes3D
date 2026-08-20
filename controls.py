import tkinter as tk
from tkinter import ttk


class ControlPanel:
    """Window containing the rendering controls."""

    def __init__(self, master, app_controller):
        style = ttk.Style()

        if "aqua" in style.theme_names():
            style.theme_use("aqua")

        self.window = tk.Toplevel(master)
        self.window.title("3D Controls")
        self.window.geometry("300x440")

        self.app = app_controller

        def create_slider(
            text,
            from_val,
            to_val,
            initial_val,
            callback
        ):
            frame = ttk.Frame(self.window)
            frame.pack(
                fill=tk.X,
                padx=20,
                pady=(10, 2)
            )

            label_var = tk.StringVar()

            def wrapped_callback(value):
                value = float(value)
                label_var.set(
                    f"{text}: {value:.1f}"
                )
                callback(value)

            label_var.set(
                f"{text}: {initial_val:.1f}"
            )

            label = ttk.Label(
                frame,
                textvariable=label_var
            )
            label.pack(side=tk.LEFT)

            slider = ttk.Scale(
                self.window,
                from_=from_val,
                to=to_val,
                orient=tk.HORIZONTAL,
                command=wrapped_callback
            )
            slider.set(initial_val)
            slider.pack(
                fill=tk.X,
                padx=20,
                pady=(0, 5)
            )

            return slider

        self.fov_slider = create_slider(
            "Field of View (FOV)",
            10,
            150,
            self.app.fov,
            self.update_fov
        )

        self.dist_slider = create_slider(
            "Camera Distance (Zoom)",
            1.0,
            15.0,
            self.app.camera_distance,
            self.update_dist
        )

        self.speed_x_slider = create_slider(
            "Rotation Speed (X-Axis)",
            -5.0,
            5.0,
            self.app.rotation_speed_vector[0],
            self.update_speed_x
        )

        self.speed_y_slider = create_slider(
            "Rotation Speed (Y-Axis)",
            -5.0,
            5.0,
            self.app.rotation_speed_vector[1],
            self.update_speed_y
        )

        self.speed_z_slider = create_slider(
            "Rotation Speed (Z-Axis)",
            -5.0,
            5.0,
            self.app.rotation_speed_vector[2],
            self.update_speed_z
        )

    def update_fov(self, value):
        self.app.fov = float(value)

    def update_dist(self, value):
        self.app.camera_distance = float(value)

    def update_speed_x(self, value):
        self.app.rotation_speed_vector[0] = float(value)

    def update_speed_y(self, value):
        self.app.rotation_speed_vector[1] = float(value)

    def update_speed_z(self, value):
        self.app.rotation_speed_vector[2] = float(value)