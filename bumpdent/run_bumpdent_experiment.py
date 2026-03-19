#!/usr/bin/env python3
"""Run a bump/dent cue-combination experiment with per-block summary graphs."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from render_stimuli import RenderParams, render_image


def direction_label(light_y: float) -> str:
    if light_y <= -0.8:
        return "bottom"
    if light_y <= -0.4:
        return "lower"
    if light_y < 0.0:
        return "slightly lower"
    if light_y < 0.4:
        return "slightly upper"
    if light_y < 0.8:
        return "upper"
    return "top"


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    block: str
    direction_label: str
    shading_y: float | None
    shadow_y: float | None
    top_is_concave: bool = False


class BumpDentExperimentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BumpDent Experiment")

        self.session_started_at = datetime.now()
        self.participant_id = ""
        self.output_path: Path | None = None

        self.block_order = ["shading", "shadow", "combined", "conflicting"]
        self.blocks = self._build_blocks()
        self.block_index = -1
        self.trial_index = -1
        self.responses: list[dict[str, str | int | float]] = []

        self.stimulus_photo: ImageTk.PhotoImage | None = None
        self.summary_photo: ImageTk.PhotoImage | None = None

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._show_intro()

    def _configure_window(self) -> None:
        self.root.minsize(980, 760)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = max(980, int(screen_w * 0.88))
        height = max(760, int(screen_h * 0.9))
        self.root.geometry(f"{width}x{height}")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#f4f6f8")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#f4f6f8", font=("TkDefaultFont", 22, "bold"))
        style.configure("Subheader.TLabel", background="#f4f6f8", font=("TkDefaultFont", 12))
        style.configure("Body.TLabel", background="#ffffff", font=("TkDefaultFont", 12))
        style.configure("Status.TLabel", background="#f4f6f8", font=("TkDefaultFont", 11))
        style.configure("Primary.TButton", font=("TkDefaultFont", 12, "bold"), padding=(14, 10))
        style.configure("Choice.TButton", font=("TkDefaultFont", 12, "bold"), padding=(14, 16))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.main = ttk.Frame(self.root, padding=24, style="App.TFrame")
        self.main.grid(row=0, column=0, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(2, weight=1)

        self.header_var = tk.StringVar(value="Bump / Dent Cue Study")
        self.subheader_var = tk.StringVar(value="Press Start when ready.")
        self.status_var = tk.StringVar(value="")

        ttk.Label(self.main, textvariable=self.header_var, style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(self.main, textvariable=self.subheader_var, style="Subheader.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 18)
        )

        self.body = ttk.Frame(self.main, style="App.TFrame")
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        ttk.Label(self.main, textvariable=self.status_var, style="Status.TLabel").grid(
            row=3, column=0, sticky="w", pady=(16, 0)
        )

        self.root.bind("<KeyPress-c>", lambda _event: self._record_response("convex"))
        self.root.bind("<KeyPress-v>", lambda _event: self._record_response("concave"))

    def _build_output_path(self) -> Path:
        output_dir = Path(__file__).resolve().parent / "data"
        output_dir.mkdir(exist_ok=True)
        timestamp = self.session_started_at.strftime("%Y%m%d_%H%M%S")
        participant_slug = self.participant_id.strip().replace(" ", "_")
        return output_dir / f"{participant_slug}_bumpdent_{timestamp}.csv"

    def _build_blocks(self) -> dict[str, list[TrialSpec]]:
        y_values = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
        blocks: dict[str, list[TrialSpec]] = {name: [] for name in self.block_order}
        trial_id = 1

        for light_y in y_values:
            label = direction_label(light_y)
            blocks["shading"].append(
                TrialSpec(trial_id=trial_id, block="shading", direction_label=label, shading_y=light_y, shadow_y=None)
            )
            trial_id += 1
            blocks["shadow"].append(
                TrialSpec(trial_id=trial_id, block="shadow", direction_label=label, shading_y=None, shadow_y=light_y)
            )
            trial_id += 1
            blocks["combined"].append(
                TrialSpec(trial_id=trial_id, block="combined", direction_label=label, shading_y=light_y, shadow_y=light_y)
            )
            trial_id += 1
            blocks["conflicting"].append(
                TrialSpec(
                    trial_id=trial_id,
                    block="conflicting",
                    direction_label=label,
                    shading_y=light_y,
                    shadow_y=-light_y,
                )
            )
            trial_id += 1

        for trials in blocks.values():
            random.shuffle(trials)
        return blocks

    def _clear_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def _show_intro(self) -> None:
        self._clear_body()
        self.header_var.set("Bump / Dent Cue Study")
        self.subheader_var.set("Judge the top stimulus only.")
        self.status_var.set(f"{sum(len(v) for v in self.blocks.values())} trials across 4 blocks")

        intro = ttk.Frame(self.body, padding=24, style="Card.TFrame")
        intro.grid(row=0, column=0, sticky="n")
        intro.columnconfigure(0, weight=1)

        ttk.Label(intro, text="Top stimulus: convex or concave?", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(intro, text="Keys: C = convex, V = concave", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        participant_row = ttk.Frame(intro, style="Card.TFrame")
        participant_row.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        ttk.Label(participant_row, text="Participant ID", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        self.participant_entry = ttk.Entry(participant_row, width=24)
        self.participant_entry.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Button(intro, text="Start", command=self._start_experiment, style="Primary.TButton").grid(
            row=3, column=0, sticky="w", pady=(20, 0)
        )

    def _start_experiment(self) -> None:
        participant = self.participant_entry.get().strip()
        if not participant:
            participant = "participant"
        self.participant_id = participant
        self.output_path = self._build_output_path()
        self.block_index = 0
        self.trial_index = 0
        self._show_trial()

    def _current_block_name(self) -> str:
        return self.block_order[self.block_index]

    def _current_trials(self) -> list[TrialSpec]:
        return self.blocks[self._current_block_name()]

    def _trial_params(self, trial: TrialSpec) -> RenderParams:
        params = RenderParams(
            width=520,
            height=560,
            radius=92.0,
            vertical_gap=42.0,
            top_is_concave=trial.top_is_concave,
            bump_strength=1.0,
            dent_strength=1.0,
            ambient=0.28,
            diffuse=0.82,
            specular=0.18,
            shininess=20.0,
            use_cosine_falloff=False,
            shadow_enabled=False,
            light_x=0.0,
            light_y=0.0,
            light_z=1.0,
            shadow_strength=0.45,
            shadow_softness=0.9,
            shadow_distance=45.0,
        )

        if trial.shading_y is not None:
            params.light_x = 0.0
            params.light_y = float(trial.shading_y)
            params.light_z = 1.0

        if trial.block == "shadow":
            params.ambient = 1.0
            params.diffuse = 0.0
            params.specular = 0.0
            params.shadow_enabled = True
            params.shadow_x = 0.0
            params.shadow_y = float(trial.shadow_y)
        elif trial.block == "combined":
            params.shadow_enabled = True
            params.shadow_x = 0.0
            params.shadow_y = float(trial.shadow_y)
        elif trial.block == "conflicting":
            params.shadow_enabled = True
            params.shadow_x = 0.0
            params.shadow_y = float(trial.shadow_y)

        return params

    def _show_trial(self) -> None:
        trials = self._current_trials()
        if self.trial_index >= len(trials):
            self._show_block_summary(self._current_block_name())
            return

        trial = trials[self.trial_index]
        self._clear_body()
        self.header_var.set("Top stimulus: convex or concave?")
        self.subheader_var.set("Judge the top stimulus.")
        self.status_var.set(
            f"Block {self.block_index + 1}/4: {trial.block} | Trial {self.trial_index + 1}/{len(trials)}"
        )

        frame = ttk.Frame(self.body, style="App.TFrame")
        frame.grid(row=0, column=0, sticky="n")
        frame.columnconfigure(0, weight=1)

        self.stimulus_photo = ImageTk.PhotoImage(render_image(self._trial_params(trial)))
        image_label = tk.Label(frame, image=self.stimulus_photo, bd=0, highlightthickness=0, bg="#f4f6f8")
        image_label.grid(row=0, column=0, pady=(0, 18))

        button_row = ttk.Frame(frame, style="App.TFrame")
        button_row.grid(row=1, column=0)
        ttk.Button(
            button_row,
            text="Convex (C)",
            command=lambda: self._record_response("convex"),
            style="Choice.TButton",
        ).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(
            button_row,
            text="Concave (V)",
            command=lambda: self._record_response("concave"),
            style="Choice.TButton",
        ).grid(row=0, column=1)

    def _record_response(self, response: str) -> None:
        if self.block_index < 0:
            return
        trials = self._current_trials()
        if self.trial_index < 0 or self.trial_index >= len(trials):
            return

        trial = trials[self.trial_index]
        self.responses.append(
            {
                "participant_id": self.participant_id,
                "trial_id": trial.trial_id,
                "block": trial.block,
                "block_index": self.block_index + 1,
                "trial_in_block": self.trial_index + 1,
                "direction_label": trial.direction_label,
                "shading_y": "" if trial.shading_y is None else trial.shading_y,
                "shadow_y": "" if trial.shadow_y is None else trial.shadow_y,
                "top_true_shape": "convex",
                "response": response,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write_responses()
        self.trial_index += 1
        self._show_trial()

    def _write_responses(self) -> None:
        if self.output_path is None:
            return
        fieldnames = [
            "participant_id",
            "trial_id",
            "block",
            "block_index",
            "trial_in_block",
            "direction_label",
            "shading_y",
            "shadow_y",
            "top_true_shape",
            "response",
            "timestamp",
        ]
        with self.output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.responses)

    def _show_block_summary(self, block_name: str) -> None:
        self._clear_body()
        self.header_var.set("Block Summary")
        self.subheader_var.set("Convex vs concave responses by light direction.")
        self.status_var.set(f"Completed block {self.block_index + 1} of 4")

        summary = ttk.Frame(self.body, style="App.TFrame")
        summary.grid(row=0, column=0, sticky="n")

        self.summary_photo = ImageTk.PhotoImage(self._build_block_chart(block_name))
        summary_label = tk.Label(summary, image=self.summary_photo, bd=0, highlightthickness=0, bg="#f4f6f8")
        summary_label.grid(row=0, column=0, pady=(0, 16))

        next_label = "Show final comparison" if self.block_index == len(self.block_order) - 1 else "Next block"
        ttk.Button(summary, text=next_label, command=self._advance_after_summary, style="Primary.TButton").grid(
            row=1, column=0, sticky="w"
        )

    def _advance_after_summary(self) -> None:
        if self.block_index == len(self.block_order) - 1:
            self._show_final_summary()
            return
        self.block_index += 1
        self.trial_index = 0
        self._show_trial()

    def _show_final_summary(self) -> None:
        self._clear_body()
        self.header_var.set("Final Summary")
        self.subheader_var.set("Compare convex-response rates across all four conditions.")
        self.status_var.set("Experiment complete")

        done = ttk.Frame(self.body, style="App.TFrame")
        done.grid(row=0, column=0, sticky="n")

        self.summary_photo = ImageTk.PhotoImage(self._build_comparison_chart())
        summary_label = tk.Label(done, image=self.summary_photo, bd=0, highlightthickness=0, bg="#f4f6f8")
        summary_label.grid(row=0, column=0, pady=(0, 16))
        ttk.Label(done, text=f"Saved responses to {self.output_path}", style="Body.TLabel").grid(
            row=1, column=0, sticky="w"
        )

    def _responses_for_block(self, block_name: str) -> list[dict[str, str | int | float]]:
        return [response for response in self.responses if response["block"] == block_name]

    def _build_block_chart(self, block_name: str) -> Image.Image:
        width, height = 920, 420
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)

        draw.text((30, 20), f"{block_name.title()} block", fill="#111111")

        left = 70
        bottom = height - 60
        top = 70
        chart_height = bottom - top
        chart_width = width - left - 40

        draw.line((left, top, left, bottom), fill="#444444", width=2)
        draw.line((left, bottom, width - 40, bottom), fill="#444444", width=2)

        directions = [direction_label(light_y) for light_y in [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]]
        data = {label: {"convex": 0, "concave": 0} for label in directions}
        for response in self._responses_for_block(block_name):
            bucket = data[str(response["direction_label"])]
            bucket[str(response["response"])] += 1

        max_count = max(1, max(bucket["convex"] + bucket["concave"] for bucket in data.values()))
        group_width = chart_width / len(directions)
        bar_width = group_width * 0.28

        for index, label in enumerate(directions):
            x_center = left + group_width * (index + 0.5)
            convex_count = data[label]["convex"]
            concave_count = data[label]["concave"]

            convex_height = chart_height * (convex_count / max_count)
            concave_height = chart_height * (concave_count / max_count)

            draw.rectangle(
                (x_center - bar_width - 4, bottom - convex_height, x_center - 4, bottom),
                fill="#2b6cb0",
            )
            draw.rectangle(
                (x_center + 4, bottom - concave_height, x_center + bar_width + 4, bottom),
                fill="#c05621",
            )
            draw.text((x_center - 28, bottom + 10), label, fill="#222222")

        draw.rectangle((width - 220, 18, width - 30, 62), outline="#dddddd")
        draw.rectangle((width - 205, 28, width - 185, 48), fill="#2b6cb0")
        draw.text((width - 178, 28), "convex", fill="#111111")
        draw.rectangle((width - 115, 28, width - 95, 48), fill="#c05621")
        draw.text((width - 88, 28), "concave", fill="#111111")

        return image

    def _build_comparison_chart(self) -> Image.Image:
        width, height = 980, 460
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)

        draw.text((30, 20), "Convex response rate by block", fill="#111111")

        left = 70
        bottom = height - 60
        top = 70
        chart_height = bottom - top
        chart_width = width - left - 40
        draw.line((left, top, left, bottom), fill="#444444", width=2)
        draw.line((left, bottom, width - 40, bottom), fill="#444444", width=2)

        directions = [direction_label(light_y) for light_y in [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]]
        block_colors = {
            "shading": "#2b6cb0",
            "shadow": "#c05621",
            "combined": "#2f855a",
            "conflicting": "#805ad5",
        }

        group_width = chart_width / len(directions)
        bar_width = group_width * 0.16
        block_names = self.block_order

        for index, label in enumerate(directions):
            x_start = left + group_width * index + group_width * 0.1
            for block_offset, block_name in enumerate(block_names):
                block_responses = [
                    response
                    for response in self._responses_for_block(block_name)
                    if response["direction_label"] == label
                ]
                total = max(1, len(block_responses))
                convex_rate = sum(1 for response in block_responses if response["response"] == "convex") / total
                bar_height = chart_height * convex_rate
                x0 = x_start + block_offset * (bar_width + 6)
                x1 = x0 + bar_width
                draw.rectangle((x0, bottom - bar_height, x1, bottom), fill=block_colors[block_name])
            draw.text((left + group_width * index + 10, bottom + 10), label, fill="#222222")

        legend_x = width - 250
        legend_y = 18
        for offset, block_name in enumerate(block_names):
            y = legend_y + offset * 24
            draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=block_colors[block_name])
            draw.text((legend_x + 28, y), block_name, fill="#111111")

        return image


def main() -> None:
    root = tk.Tk()
    BumpDentExperimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
