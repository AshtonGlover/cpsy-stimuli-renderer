#!/usr/bin/env python3
"""Interactive convex/concave bump renderer — smooth bump/dent embedded in a flat surface."""

import math
import tkinter as tk
from tkinter import ttk


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

def vec_from_angles(azimuth_deg: float, elevation_deg: float):
    """Return a unit vector from azimuth/elevation in degrees."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    c = math.cos(el)
    x = c * math.cos(az)
    y = c * math.sin(az)
    z = math.sin(el)
    mag = math.sqrt(x * x + y * y + z * z)
    if mag == 0:
        return (0.0, 0.0, 1.0)
    return (x / mag, y / mag, z / mag)

def normalize(x, y, z):
    mag = math.sqrt(x*x + y*y + z*z)
    if mag < 1e-9:
        return (0.0, 0.0, 1.0)
    return (x/mag, y/mag, z/mag)


class BumpRendererGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Convex/Concave Bump Renderer")

        self.width = 360
        self.height = 360

        self.shape_var = tk.StringVar(value="convex")

        self.diffuse_az_var = tk.DoubleVar(value=315.0)
        self.diffuse_el_var = tk.DoubleVar(value=45.0)

        self.shadow_az_var = tk.DoubleVar(value=220.0)
        self.shadow_el_var = tk.DoubleVar(value=30.0)

        self.ambient_var = tk.DoubleVar(value=0.25)
        self.diffuse_strength_var = tk.DoubleVar(value=0.9)

        self.bg_gray_var = tk.IntVar(value=165)
        self.sphere_albedo_var = tk.IntVar(value=220)

        self.shadow_strength_var = tk.DoubleVar(value=0.45)
        self.shadow_softness_var = tk.DoubleVar(value=0.9)
        self.shadow_distance_var = tk.DoubleVar(value=45.0)

        self.radius_var = tk.DoubleVar(value=95.0)

        # Height scale: how "tall" the bump is relative to its radius.
        # 1.0 ≈ hemisphere; 0.2 ≈ very shallow disc.
        self.height_scale_var = tk.DoubleVar(value=0.35)

        self._build_ui()
        self.render()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        preview = ttk.Frame(outer)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.photo = tk.PhotoImage(width=self.width, height=self.height)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        shape_labelframe = ttk.LabelFrame(controls, text="Shape")
        shape_labelframe.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            shape_labelframe,
            text="Convex (bump)",
            value="convex",
            variable=self.shape_var,
            command=self.render,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            shape_labelframe,
            text="Concave (dent)",
            value="concave",
            variable=self.shape_var,
            command=self.render,
        ).grid(row=1, column=0, sticky="w")

        self._add_slider_group(
            controls,
            "Diffuse Light",
            start_row=1,
            sliders=[
                ("Azimuth", self.diffuse_az_var, 0, 359),
                ("Elevation", self.diffuse_el_var, 1, 89),
                ("Ambient", self.ambient_var, 0.0, 1.0),
                ("Strength", self.diffuse_strength_var, 0.0, 1.5),
            ],
        )

        self._add_slider_group(
            controls,
            "Directional Shadow",
            start_row=2,
            sliders=[
                ("Azimuth", self.shadow_az_var, 0, 359),
                ("Elevation", self.shadow_el_var, 1, 89),
                ("Strength", self.shadow_strength_var, 0.0, 1.0),
                ("Softness", self.shadow_softness_var, 0.1, 2.5),
                ("Spread", self.shadow_distance_var, 0.0, 220.0),
            ],
        )

        self._add_slider_group(
            controls,
            "Material / Scene",
            start_row=3,
            sliders=[
                ("Background", self.bg_gray_var, 0, 255),
                ("Albedo", self.sphere_albedo_var, 20, 255),
                ("Radius", self.radius_var, 50, 130),
                ("Height", self.height_scale_var, 0.05, 1.5),
            ],
        )

        tip = ttk.Label(
            controls,
            text=(
                "Tip: set different diffuse vs shadow azimuths\n"
                "to create conflicting shape cues.\n"
                "Height controls bump depth (low = shallow)."
            ),
            justify="left",
        )
        tip.grid(row=4, column=0, sticky="w", pady=(6, 0))

    def _add_slider_group(self, parent, title, start_row, sliders):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=start_row, column=0, sticky="ew", pady=(0, 8))

        for i, (label, var, min_v, max_v) in enumerate(sliders):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w")

            scale = ttk.Scale(
                frame,
                orient="horizontal",
                from_=min_v,
                to=max_v,
                variable=var,
                command=lambda _x: self.render(),
            )
            scale.grid(row=i, column=1, sticky="ew", padx=4)

            value_label = ttk.Label(frame, width=6)
            value_label.grid(row=i, column=2, sticky="e")

            def update_label(*_args, v=var, l=value_label):
                raw = v.get()
                if abs(raw - round(raw)) < 1e-6:
                    l.config(text=f"{int(round(raw))}")
                else:
                    l.config(text=f"{raw:.2f}")

            var.trace_add("write", update_label)
            update_label()

        frame.columnconfigure(1, weight=1)

    def render(self):
        w, h = self.width, self.height
        cx, cy = w * 0.5, h * 0.5

        radius = self.radius_var.get()
        height_scale = self.height_scale_var.get()
        # Maximum bump height in the same units as radius
        max_z = radius * height_scale

        bg_gray = int(self.bg_gray_var.get())
        sphere_albedo = int(self.sphere_albedo_var.get())

        diffuse_light = vec_from_angles(self.diffuse_az_var.get(), self.diffuse_el_var.get())
        shadow_light = vec_from_angles(self.shadow_az_var.get(), self.shadow_el_var.get())

        ambient = clamp01(self.ambient_var.get())
        diffuse_strength = max(0.0, self.diffuse_strength_var.get())

        shadow_strength = clamp01(self.shadow_strength_var.get())
        shadow_softness = max(0.1, self.shadow_softness_var.get())
        shadow_spread = max(0.0, self.shadow_distance_var.get())

        shape_is_concave = self.shape_var.get() == "concave"

        # Flat-surface normal (used outside the bump)
        flat_normal = (0.0, 0.0, 1.0)

        rows = []
        for py in range(h):
            row_colors = []
            y = (py + 0.5) - cy
            for px in range(w):
                x = (px + 0.5) - cx

                # --- Directional contact shadow on background ---
                shadow_factor = 0.0
                d = math.sqrt(max(x * x + y * y, 0.0))
                edge_dist = d - radius
                if edge_dist >= 0.0:
                    edge_width = radius * (0.03 + 0.17 * (shadow_softness / 2.5)) + 1.0 * shadow_spread
                    if edge_dist < edge_width:
                        t = clamp01(edge_dist / max(1e-6, edge_width))
                        falloff = 1.0 - t
                        dir_mod = 0.0
                        if d > 1e-6:
                            ndx = x / d
                            ndy = -y / d
                            light_dot = ndx * shadow_light[0] + ndy * shadow_light[1]
                            dir_mod = clamp01(-light_dot)
                        shadow_factor = shadow_strength * falloff * dir_mod

                # --- Shading ---
                r_norm = d / radius  # normalized radial distance

                if r_norm <= 1.0:
                    # --- Cosine bump height field ---
                    # h(r) = max_z * 0.5 * (1 + cos(pi * r_norm))
                    # This is 1 at center, 0 at edge, C1-continuous.
                    # dh/dr = -max_z * 0.5 * pi * sin(pi * r_norm)
                    dh_dr = -max_z * 0.5 * math.pi * math.sin(math.pi * r_norm)

                    # Surface gradient in x and y:
                    # dh/dx = dh/dr * dr/dx = dh/dr * (x / (r * radius))  [chain rule]
                    if d > 1e-9:
                        dh_dx = dh_dr * (x / (d * radius))
                        dh_dy = dh_dr * (y / (d * radius))
                    else:
                        dh_dx = 0.0
                        dh_dy = 0.0

                    # Surface normal from gradient: N = normalize(-dh/dx, -dh/dy, 1)
                    nx_raw = -dh_dx
                    ny_raw = -dh_dy
                    nz_raw = 1.0

                    if shape_is_concave:
                        # Flip the gradient to invert the bump into a dent
                        nx_raw = -nx_raw
                        ny_raw = -ny_raw

                    nx, ny, nz = normalize(nx_raw, ny_raw, nz_raw)

                    # Flip y for screen→world convention
                    ny = -ny

                    lambert = max(0.0, nx * diffuse_light[0] + ny * diffuse_light[1] + nz * diffuse_light[2])
                    lit = ambient + diffuse_strength * lambert
                    lit = clamp01(lit)
                    intensity = sphere_albedo * lit
                else:
                    # Flat background with contact shadow
                    flat_lambert = max(0.0, flat_normal[2] * diffuse_light[2])
                    lit = ambient + diffuse_strength * flat_lambert
                    lit = clamp01(lit)
                    base = bg_gray * lit * (1.0 - shadow_factor)
                    intensity = base

                g = int(max(0, min(255, round(intensity))))
                row_colors.append(f"#{g:02x}{g:02x}{g:02x}")
            rows.append("{" + " ".join(row_colors) + "}")

        self.photo.put(" ".join(rows), to=(0, 0, w, h))

def main():
    root = tk.Tk()
    app = BumpRendererGUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()