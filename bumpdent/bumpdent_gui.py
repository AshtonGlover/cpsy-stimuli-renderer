#!/usr/bin/env python3
"""Interactive bump/dent stimulus renderer."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import ImageTk

from render_stimuli import RenderParams, clamp, render_image, render_side_profile


class BumpDentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bump / Dent Stimulus Renderer")

        self.width = 420
        self.height = 520

        self.background_var = tk.IntVar(value=182)
        self.albedo_var = tk.IntVar(value=196)
        self.radius_var = tk.DoubleVar(value=96.0)
        self.vertical_gap_var = tk.DoubleVar(value=38.0)
        self.top_is_concave_var = tk.BooleanVar(value=False)
        self.center_x_var = tk.DoubleVar(value=0.0)
        self.center_y_var = tk.DoubleVar(value=0.0)
        self.bump_strength_var = tk.DoubleVar(value=0.95)
        self.dent_strength_var = tk.DoubleVar(value=1.25)
        self.light_x_var = tk.DoubleVar(value=0.0)
        self.light_y_var = tk.DoubleVar(value=0.75)
        self.light_z_var = tk.DoubleVar(value=1.1)
        self.ambient_var = tk.DoubleVar(value=0.28)
        self.diffuse_var = tk.DoubleVar(value=0.82)
        self.specular_var = tk.DoubleVar(value=0.18)
        self.shininess_var = tk.DoubleVar(value=20.0)
        self.cosine_falloff_var = tk.BooleanVar(value=False)
        self.flat_profile_var = tk.BooleanVar(value=True)
        self.shadow_enabled_var = tk.BooleanVar(value=True)
        self.shadow_follows_light_y_var = tk.BooleanVar(value=True)
        self.shadow_azimuth_var = tk.DoubleVar(value=220.0)
        self.shadow_elevation_var = tk.DoubleVar(value=30.0)
        self.shadow_strength_var = tk.DoubleVar(value=0.45)
        self.shadow_softness_var = tk.DoubleVar(value=0.9)
        self.shadow_distance_var = tk.DoubleVar(value=45.0)

        self.photo = None
        self.side_profile_photo = None
        self.shadow_window = None
        self.side_profile_window = None
        self._build_ui()
        self.render()

    def _build_ui(self) -> None:
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
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")

        lighting_frame = self._add_slider_group(
            controls,
            "Lighting",
            0,
            [
                ("Light Y", self.light_y_var, -1.0, 1.0),
                ("Light X", self.light_x_var, -1.0, 1.0),
                ("Light Z", self.light_z_var, 0.1, 2.0),
                ("Ambient", self.ambient_var, 0.0, 1.2),
                ("Diffuse", self.diffuse_var, 0.0, 1.5),
                ("Specular", self.specular_var, 0.0, 0.8),
                ("Shininess", self.shininess_var, 2.0, 80.0),
            ],
        )
        ttk.Button(lighting_frame, text="Set Ambiguous", command=self.set_ambiguous_lighting).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )
        ttk.Button(lighting_frame, text="Restore Default", command=self.restore_default_lighting).grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )

        self._add_slider_group(
            controls,
            "Stimuli",
            1,
            [
                ("Radius", self.radius_var, 40.0, 150.0),
                ("Gap", self.vertical_gap_var, -100.0, 120.0),
                ("Center X", self.center_x_var, -120.0, 120.0),
                ("Center Y", self.center_y_var, -100.0, 100.0),
                ("Bump", self.bump_strength_var, 0.2, 1.6),
                ("Dent", self.dent_strength_var, 0.2, 1.6),
            ],
        )

        self._add_slider_group(
            controls,
            "Scene",
            2,
            [
                ("Background", self.background_var, 120, 230),
                ("Albedo", self.albedo_var, 120, 245),
            ],
        )

        orientation_frame = ttk.LabelFrame(controls, text="Orientation")
        orientation_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(
            orientation_frame,
            text="Top convex / bottom concave",
            value=False,
            variable=self.top_is_concave_var,
            command=self.render,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            orientation_frame,
            text="Top concave / bottom convex",
            value=True,
            variable=self.top_is_concave_var,
            command=self.render,
        ).grid(row=1, column=0, sticky="w")

        options_frame = ttk.LabelFrame(controls, text="Options")
        options_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Checkbutton(
            options_frame,
            text="Cosine edge falloff",
            variable=self.cosine_falloff_var,
            command=self.render,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            options_frame,
            text="Flat embedded profile",
            variable=self.flat_profile_var,
            command=self.render,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            options_frame,
            text="Directional shadows",
            variable=self.shadow_enabled_var,
            command=self.render,
        ).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(
            options_frame,
            text="Shadows follow Light Y",
            variable=self.shadow_follows_light_y_var,
            command=self.render,
        ).grid(row=3, column=0, sticky="w")

        button_row = ttk.Frame(controls)
        button_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_row, text="Export PNG", command=self.export_png).grid(row=0, column=0, sticky="ew")
        ttk.Button(button_row, text="Shadow Controls", command=self.open_shadow_window).grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(button_row, text="Side Profile", command=self.open_side_profile_window).grid(
            row=2, column=0, sticky="ew", pady=(6, 0)
        )

        tip = ttk.Label(
            controls,
            text="Light Y controls the cue flip: -1 is bottom light, 1 is top light.",
            justify="left",
            wraplength=260,
        )
        tip.grid(row=6, column=0, sticky="w", pady=(8, 0))

    def _format_value(self, variable) -> str:
        value = variable.get()
        if isinstance(variable, tk.IntVar):
            return f"{int(round(value))}"
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}"
        return f"{value:.2f}"

    def _add_slider_group(self, parent, title, row, sliders):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))

        for index, (label, variable, low, high) in enumerate(sliders):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w")
            slider = ttk.Scale(
                frame,
                orient="horizontal",
                from_=low,
                to=high,
                variable=variable,
                command=lambda _value: self.render(),
            )
            slider.grid(row=index, column=1, sticky="ew", padx=4)

            entry_var = tk.StringVar(value=self._format_value(variable))
            entry = ttk.Entry(frame, textvariable=entry_var, width=8)
            entry.grid(row=index, column=2, sticky="e")

            def update_entry(*_args, var=variable, entry_string=entry_var):
                entry_string.set(self._format_value(var))

            def apply_entry(
                _event=None,
                var=variable,
                entry_string=entry_var,
                min_value=low,
                max_value=high,
            ):
                raw = entry_string.get().strip()
                try:
                    parsed = float(raw)
                except ValueError:
                    entry_string.set(self._format_value(var))
                    return

                if isinstance(var, tk.IntVar):
                    value = int(round(clamp(parsed, min_value, max_value)))
                else:
                    value = clamp(parsed, min_value, max_value)
                var.set(value)
                entry_string.set(self._format_value(var))
                self.render()

            variable.trace_add("write", update_entry)
            entry.bind("<Return>", apply_entry)
            entry.bind("<FocusOut>", apply_entry)
            update_entry()

        frame.columnconfigure(1, weight=1)
        return frame

    def _params(self) -> RenderParams:
        light_y = clamp(self.light_y_var.get(), -1.0, 1.0)
        shadow_follows_light_y = self.shadow_follows_light_y_var.get()

        return RenderParams(
            width=self.width,
            height=self.height,
            background=int(clamp(self.background_var.get(), 0, 255)),
            albedo=int(clamp(self.albedo_var.get(), 0, 255)),
            radius=max(20.0, self.radius_var.get()),
            vertical_gap=self.vertical_gap_var.get(),
            top_is_concave=self.top_is_concave_var.get(),
            center_x=self.center_x_var.get(),
            center_y=self.center_y_var.get(),
            bump_strength=max(0.2, self.bump_strength_var.get()),
            dent_strength=max(0.2, self.dent_strength_var.get()),
            light_x=clamp(self.light_x_var.get(), -1.0, 1.0),
            light_y=light_y,
            light_z=max(0.1, self.light_z_var.get()),
            ambient=max(0.0, self.ambient_var.get()),
            diffuse=max(0.0, self.diffuse_var.get()),
            specular=max(0.0, self.specular_var.get()),
            shininess=max(1.0, self.shininess_var.get()),
            use_cosine_falloff=self.cosine_falloff_var.get(),
            use_flat_profile=self.flat_profile_var.get(),
            shadow_enabled=self.shadow_enabled_var.get(),
            shadow_x=0.0 if shadow_follows_light_y else None,
            shadow_y=light_y if shadow_follows_light_y else None,
            shadow_azimuth=self.shadow_azimuth_var.get() % 360.0,
            shadow_elevation=clamp(self.shadow_elevation_var.get(), 1.0, 89.0),
            shadow_strength=clamp(self.shadow_strength_var.get(), 0.0, 1.0),
            shadow_softness=max(0.1, self.shadow_softness_var.get()),
            shadow_distance=max(0.0, self.shadow_distance_var.get()),
        )

    def set_ambiguous_lighting(self) -> None:
        self.light_x_var.set(0.0)
        self.light_y_var.set(0.0)
        self.light_z_var.set(1.0)
        self.ambient_var.set(1.0)
        self.diffuse_var.set(0.0)
        self.specular_var.set(0.0)
        self.render()

    def restore_default_lighting(self) -> None:
        self.top_is_concave_var.set(False)
        self.light_x_var.set(0.0)
        self.light_y_var.set(0.75)
        self.light_z_var.set(1.1)
        self.ambient_var.set(0.28)
        self.diffuse_var.set(0.82)
        self.specular_var.set(0.18)
        self.shininess_var.set(20.0)
        self.render()

    def open_shadow_window(self) -> None:
        if self.shadow_window is not None and self.shadow_window.winfo_exists():
            self.shadow_window.lift()
            self.shadow_window.focus_force()
            return

        self.shadow_window = tk.Toplevel(self.root)
        self.shadow_window.title("Shadow Controls")
        self.shadow_window.transient(self.root)
        self.shadow_window.columnconfigure(0, weight=1)

        frame = ttk.Frame(self.shadow_window, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            frame,
            text="Enable directional shadows",
            variable=self.shadow_enabled_var,
            command=self.render,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self._add_slider_group(
            frame,
            "Directional Shadow",
            1,
            [
                ("Azimuth", self.shadow_azimuth_var, 0.0, 359.0),
                ("Elevation", self.shadow_elevation_var, 1.0, 89.0),
                ("Strength", self.shadow_strength_var, 0.0, 1.0),
                ("Softness", self.shadow_softness_var, 0.1, 2.5),
                ("Spread", self.shadow_distance_var, 0.0, 220.0),
            ],
        )

        ttk.Label(
            frame,
            text="Turn off 'Shadows follow Light Y' to edit shadow direction independently.",
            justify="left",
            wraplength=260,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.shadow_window.protocol("WM_DELETE_WINDOW", self._close_shadow_window)

    def _close_shadow_window(self) -> None:
        if self.shadow_window is not None:
            self.shadow_window.destroy()
            self.shadow_window = None

    def open_side_profile_window(self) -> None:
        if self.side_profile_window is not None and self.side_profile_window.winfo_exists():
            self.side_profile_window.lift()
            self.side_profile_window.focus_force()
            self._render_side_profile()
            return

        self.side_profile_window = tk.Toplevel(self.root)
        self.side_profile_window.title("Side Profile")
        self.side_profile_window.transient(self.root)
        self.side_profile_window.columnconfigure(0, weight=1)
        self.side_profile_window.rowconfigure(0, weight=1)

        container = ttk.Frame(self.side_profile_window, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.side_profile_label = tk.Label(container, bd=0, highlightthickness=0, bg="#f4f6f8")
        self.side_profile_label.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            container,
            text="Cross-section preview from the side using the current bump/dent settings.",
            justify="left",
            wraplength=420,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.side_profile_window.protocol("WM_DELETE_WINDOW", self._close_side_profile_window)
        self._render_side_profile()

    def _render_side_profile(self) -> None:
        if self.side_profile_window is None or not self.side_profile_window.winfo_exists():
            return
        image = render_side_profile(self._params())
        self.side_profile_photo = ImageTk.PhotoImage(image)
        self.side_profile_label.configure(image=self.side_profile_photo)

    def _close_side_profile_window(self) -> None:
        if self.side_profile_window is not None:
            self.side_profile_window.destroy()
            self.side_profile_window = None

    def render(self) -> None:
        image = render_image(self._params())
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)
        self._render_side_profile()

    def export_png(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export stimulus PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="bumpdent_stimulus.png",
        )
        if not path:
            return
        render_image(self._params()).save(path)


def main() -> None:
    root = tk.Tk()
    BumpDentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
