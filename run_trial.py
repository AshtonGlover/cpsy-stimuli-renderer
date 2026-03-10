#!/usr/bin/env python3
"""Run a brief convex/concave rating experiment."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def vec_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
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


def azimuth_label(azimuth_deg: float) -> str:
    labels = {
        0.0: "right",
        60.0: "upper-right",
        120.0: "upper-left",
        180.0: "left",
        240.0: "lower-left",
        300.0: "lower-right",
    }
    return labels[azimuth_deg % 360.0]


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    shape: str
    diffuse_direction: str
    shadow_direction: str
    diffuse_azimuth: float
    shadow_azimuth: float
    elevation: float
    ambient: float
    diffuse_strength: float
    shadow_strength: float
    shadow_softness: float
    shadow_distance: float


class StimulusRenderer:
    def __init__(self, width: int = 360, height: int = 360):
        self.width = width
        self.height = height
        self.radius = 95.0
        self.bg_gray = 165
        self.sphere_albedo = 220

    def render_to_photoimage(self, trial: TrialSpec) -> tk.PhotoImage:
        photo = tk.PhotoImage(width=self.width, height=self.height)
        rows = self._build_rows(trial)
        photo.put(" ".join(rows), to=(0, 0, self.width, self.height))
        return photo

    def _build_rows(self, trial: TrialSpec) -> list[str]:
        w, h = self.width, self.height
        cx, cy = w * 0.5, h * 0.5
        radius = self.radius
        r2 = radius * radius

        diffuse_light = vec_from_angles(trial.diffuse_azimuth, trial.elevation)
        shadow_light = vec_from_angles(trial.shadow_azimuth, trial.elevation)
        shape_is_concave = trial.shape == "concave"

        rows: list[str] = []
        for py in range(h):
            row_colors = []
            y = (py + 0.5) - cy
            for px in range(w):
                x = (px + 0.5) - cx

                shadow_factor = 0.0
                d = math.sqrt(max(x * x + y * y, 0.0))
                edge_dist = d - radius
                if edge_dist >= 0.0:
                    edge_width = radius * (0.03 + 0.17 * (trial.shadow_softness / 2.5)) + trial.shadow_distance
                    if edge_dist < edge_width:
                        t = clamp01(edge_dist / max(1e-6, edge_width))
                        falloff = 1.0 - t
                        dir_mod = 0.0
                        if d > 1e-6:
                            ndx = x / d
                            ndy = -y / d
                            light_dot = ndx * shadow_light[0] + ndy * shadow_light[1]
                            dir_mod = clamp01(-light_dot)
                        shadow_factor = trial.shadow_strength * falloff * dir_mod

                base = self.bg_gray * (1.0 - shadow_factor)
                intensity = base

                dsq = x * x + y * y
                if dsq <= r2:
                    nx = x / radius
                    ny = -y / radius
                    nz_sq = max(0.0, 1.0 - nx * nx - ny * ny)
                    nz = math.sqrt(nz_sq)

                    if shape_is_concave:
                        nx = -nx
                        ny = -ny

                    lambert = max(0.0, nx * diffuse_light[0] + ny * diffuse_light[1] + nz * diffuse_light[2])
                    lit = clamp01(trial.ambient + trial.diffuse_strength * lambert)
                    intensity = self.sphere_albedo * lit

                g = int(max(0, min(255, round(intensity))))
                row_colors.append(f"#{g:02x}{g:02x}{g:02x}")
            rows.append("{" + " ".join(row_colors) + "}")
        return rows


class TrialRunnerApp:
    SCALE_LABELS = {
        1: "convex",
        2: "mostly convex",
        3: "both",
        4: "mostly concave",
        5: "concave",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Convex Concave Trial")

        self._configure_window()
        self._configure_styles()

        self.renderer = StimulusRenderer(width=self._stimulus_size(), height=self._stimulus_size())
        self.trials = self._build_trials()
        self.trial_index = -1
        self.current_photo: tk.PhotoImage | None = None
        self.responses: list[dict[str, str | int | float]] = []
        self.session_started_at = datetime.now()
        self.participant_id = ""
        self.output_path: Path | None = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.main = ttk.Frame(self.root, padding=24, style="App.TFrame")
        self.main.grid(row=0, column=0, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(2, weight=1)

        self.header_var = tk.StringVar(value="Convex/Concave Study")
        self.subheader_var = tk.StringVar(value="Press Start when ready.")
        self.status_var = tk.StringVar(value="")

        ttk.Label(self.main, textvariable=self.header_var, style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(self.main, textvariable=self.subheader_var, style="Subheader.TLabel").grid(
            row=1, column=0, sticky="nw", pady=(6, 18)
        )

        self.body = ttk.Frame(self.main, style="App.TFrame")
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        ttk.Label(self.main, textvariable=self.status_var, style="Status.TLabel").grid(
            row=3, column=0, sticky="w", pady=(16, 0)
        )

        self.root.bind("<KeyPress>", self._on_keypress)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._exit_fullscreen)
        self._show_intro()

    def _configure_window(self):
        self.root.minsize(960, 720)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = max(960, int(screen_w * 0.88))
        height = max(720, int(screen_h * 0.9))
        self.root.geometry(f"{width}x{height}")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#f4f6f8")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Header.TLabel", background="#f4f6f8", font=("TkDefaultFont", 22, "bold"))
        style.configure("Subheader.TLabel", background="#f4f6f8", font=("TkDefaultFont", 12))
        style.configure("Body.TLabel", background="#ffffff", font=("TkDefaultFont", 13))
        style.configure("Status.TLabel", background="#f4f6f8", font=("TkDefaultFont", 11))
        style.configure("Error.TLabel", background="#ffffff", foreground="#b00020", font=("TkDefaultFont", 11))
        style.configure("Primary.TButton", font=("TkDefaultFont", 12, "bold"), padding=(16, 10))
        style.configure("Scale.TButton", font=("TkDefaultFont", 12, "bold"), padding=(12, 18))

    def _stimulus_size(self) -> int:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        return max(360, min(680, int(min(screen_w * 0.42, screen_h * 0.55))))

    def _build_output_path(self) -> Path:
        output_dir = Path(__file__).resolve().parent / "data"
        output_dir.mkdir(exist_ok=True)
        timestamp = self.session_started_at.strftime("%Y%m%d_%H%M%S")
        participant_slug = self.participant_id.strip().replace(" ", "_")
        return output_dir / f"{participant_slug}_{timestamp}.csv"

    def _build_trials(self) -> list[TrialSpec]:
        azimuths = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        trials: list[TrialSpec] = []
        trial_id = 1
        for diffuse_azimuth in azimuths:
            for shadow_azimuth in azimuths:
                trials.append(
                    TrialSpec(
                        trial_id=trial_id,
                        shape="convex",
                        diffuse_direction=azimuth_label(diffuse_azimuth),
                        shadow_direction=azimuth_label(shadow_azimuth),
                        diffuse_azimuth=diffuse_azimuth,
                        shadow_azimuth=shadow_azimuth,
                        elevation=45.0,
                        ambient=0.30,
                        diffuse_strength=0.72,
                        shadow_strength=0.36,
                        shadow_softness=1.1,
                        shadow_distance=48.0,
                    )
                )
                trial_id += 1
        random.shuffle(trials)
        return trials

    def _clear_body(self):
        for child in self.body.winfo_children():
            child.destroy()

    def _show_intro(self):
        self._clear_body()
        self.body.rowconfigure(0, weight=1)
        intro_wrap = ttk.Frame(self.body, style="App.TFrame")
        intro_wrap.grid(row=0, column=0, sticky="nsew")
        intro_wrap.columnconfigure(0, weight=1)
        intro_wrap.columnconfigure(1, weight=1)
        intro_wrap.rowconfigure(0, weight=1)
        intro_wrap.rowconfigure(1, weight=1)

        intro = ttk.Frame(intro_wrap, padding=28, style="Card.TFrame")
        intro.grid(row=0, column=0, columnspan=2, sticky="")
        intro.columnconfigure(0, weight=1)

        ttk.Label(intro, text="Look at each sphere and rate what you see.", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(intro, text="Enter your ID, then choose 1 to 5.", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 6)
        )
        ttk.Label(intro, text="1 = convex, 5 = concave. Use keys 1-5 or click.", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 18)
        )
        id_row = ttk.Frame(intro)
        id_row.grid(row=3, column=0, sticky="w", pady=(0, 14))
        ttk.Label(id_row, text="Participant ID").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.participant_id_var = tk.StringVar()
        id_entry = ttk.Entry(id_row, textvariable=self.participant_id_var, width=28, font=("TkDefaultFont", 12))
        id_entry.grid(row=0, column=1, sticky="w")
        id_entry.focus_set()

        self.intro_error_var = tk.StringVar(value="")
        ttk.Label(intro, textvariable=self.intro_error_var, style="Error.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, 8)
        )

        ttk.Button(intro, text="Start", command=self._start_experiment, style="Primary.TButton").grid(
            row=5, column=0, sticky="w"
        )
        self.status_var.set(f"{len(self.trials)} trials")

    def _start_experiment(self):
        participant_id = self.participant_id_var.get().strip()
        if not participant_id:
            self.intro_error_var.set("Enter a participant ID.")
            return

        self.participant_id = participant_id
        self.output_path = self._build_output_path()
        self.trial_index = 0
        self._show_trial()

    def _show_trial(self):
        if self.trial_index >= len(self.trials):
            self._finish_experiment()
            return

        self._clear_body()
        trial = self.trials[self.trial_index]

        self.header_var.set("Convex or concave?")
        self.subheader_var.set("Rate what you see.")
        self.status_var.set(f"Trial {self.trial_index + 1} of {len(self.trials)}")

        trial_frame = ttk.Frame(self.body, style="App.TFrame")
        trial_frame.grid(row=0, column=0, sticky="nsew")
        trial_frame.columnconfigure(0, weight=1)
        for column in range(5):
            trial_frame.columnconfigure(column, weight=1, uniform="scale")
        trial_frame.rowconfigure(0, weight=1)

        self.renderer = StimulusRenderer(width=self._stimulus_size(), height=self._stimulus_size())
        self.current_photo = self.renderer.render_to_photoimage(trial)
        canvas = tk.Canvas(
            trial_frame,
            width=self.renderer.width,
            height=self.renderer.height,
            bg="#f4f6f8",
            highlightthickness=0,
            bd=0,
        )
        canvas.grid(row=0, column=0, columnspan=5, pady=(0, 22))
        canvas.create_image(0, 0, anchor="nw", image=self.current_photo)

        for value in range(1, 6):
            ttk.Button(
                trial_frame,
                text=f"{value}\n{self.SCALE_LABELS[value]}",
                command=lambda choice=value: self._record_response(choice),
                style="Scale.TButton",
                width=18,
            ).grid(row=1, column=value - 1, padx=8, sticky="ew")

    def _record_response(self, response_value: int):
        if self.trial_index < 0 or self.trial_index >= len(self.trials):
            return

        trial = self.trials[self.trial_index]
        timestamp = datetime.now()
        self.responses.append(
            {
                "participant_id": self.participant_id,
                "trial_id": trial.trial_id,
                "order_index": self.trial_index + 1,
                "shape": trial.shape,
                "diffuse_direction": trial.diffuse_direction,
                "diffuse_azimuth": trial.diffuse_azimuth,
                "shadow_direction": trial.shadow_direction,
                "shadow_azimuth": trial.shadow_azimuth,
                "elevation": trial.elevation,
                "response": response_value,
                "response_label": self.SCALE_LABELS[response_value],
                "timestamp": timestamp.isoformat(timespec="seconds"),
            }
        )
        self._write_responses()
        self.trial_index += 1
        self._show_trial()

    def _write_responses(self):
        fieldnames = [
            "participant_id",
            "trial_id",
            "order_index",
            "shape",
            "diffuse_direction",
            "diffuse_azimuth",
            "shadow_direction",
            "shadow_azimuth",
            "elevation",
            "response",
            "response_label",
            "timestamp",
        ]
        if self.output_path is None:
            return
        with self.output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.responses)

    def _finish_experiment(self):
        self._clear_body()
        self.header_var.set("Finished")
        self.subheader_var.set("Thanks.")
        self.status_var.set(f"Saved to {self.output_path.name if self.output_path else 'data'}")

        done_wrap = ttk.Frame(self.body, style="App.TFrame")
        done_wrap.grid(row=0, column=0, sticky="nsew")
        done_wrap.columnconfigure(0, weight=1)
        done_wrap.columnconfigure(1, weight=1)
        done_wrap.rowconfigure(0, weight=1)
        done_wrap.rowconfigure(1, weight=1)

        done = ttk.Frame(done_wrap, padding=28, style="Card.TFrame")
        done.grid(row=0, column=0, columnspan=2, sticky="")
        done.columnconfigure(0, weight=1)
        ttk.Label(done, text="Your responses were saved.", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(done, text=str(self.output_path) if self.output_path else "", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 12)
        )
        ttk.Button(done, text="Close", command=self.root.destroy, style="Primary.TButton").grid(
            row=2, column=0, sticky="w"
        )

    def _on_keypress(self, event: tk.Event):
        if event.char in {"1", "2", "3", "4", "5"} and 0 <= self.trial_index < len(self.trials):
            self._record_response(int(event.char))

    def _toggle_fullscreen(self, _event: tk.Event | None = None):
        is_fullscreen = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not is_fullscreen)

    def _exit_fullscreen(self, _event: tk.Event | None = None):
        self.root.attributes("-fullscreen", False)


def main():
    root = tk.Tk()
    app = TrialRunnerApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
