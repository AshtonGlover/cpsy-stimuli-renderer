from __future__ import annotations

import csv
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk


CATEGORY_ORDER = ["shading", "shadow", "incongruent"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
OUTPUT_DIRNAME = "results"
LABELS_FILENAME = "labels.csv"
ROTATED_IMAGE_SUFFIX = "_rot180"
TRIAL_REPETITIONS = 20
IMAGE_DISPLAY_MS = 850
BATCH_COUNT = 4
OUTPUT_FIELDNAMES = [
    "participant_id",
    "trial_number",
    "category",
    "image_id",
    "image_name",
    "image_path",
    "rotation_degrees",
    "response",
    "timestamp",
]
MAX_IMAGE_SIZE = (900, 700)
WINDOW_SIZE = "1200x940"
APP_BG = "#eef3f1"
CARD_BG = "#ffffff"
CARD_BORDER = "#d7e2dd"
TEXT_PRIMARY = "#18322c"
TEXT_MUTED = "#4f6a63"
PRIMARY_BUTTON_BG = "#3a3f45"
PRIMARY_BUTTON_ACTIVE_BG = PRIMARY_BUTTON_BG
CONVEX_BUTTON_BG = PRIMARY_BUTTON_BG
CONVEX_BUTTON_ACTIVE_BG = CONVEX_BUTTON_BG
CONCAVE_BUTTON_BG = PRIMARY_BUTTON_BG
CONCAVE_BUTTON_ACTIVE_BG = CONCAVE_BUTTON_BG


@dataclass(frozen=True)
class Trial:
    category: str
    image_id: int
    image_path: Path
    rotation_degrees: int


@dataclass(frozen=True)
class ImageLabel:
    image_id: int
    category: str
    image_name: str
    image_path: Path
    mock_label: str


def load_image_labels(labels_path: Path, data_dir: Path) -> dict[Path, ImageLabel]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    expected_paths = {
        path.resolve()
        for category in CATEGORY_ORDER
        for path in (data_dir / category).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels_by_path: dict[Path, ImageLabel] = {}

    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"image_id", "category", "image_name", "image_path", "mock_label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{labels_path} must contain columns: {', '.join(sorted(required_columns))}"
            )

        for row in reader:
            image_path = Path(row["image_path"]).resolve()
            image_label = ImageLabel(
                image_id=int(row["image_id"]),
                category=row["category"],
                image_name=row["image_name"],
                image_path=image_path,
                mock_label=row["mock_label"],
            )
            labels_by_path[image_path] = image_label

    missing_paths = expected_paths - set(labels_by_path)
    if missing_paths:
        missing_str = ", ".join(str(path) for path in sorted(missing_paths))
        raise ValueError(f"labels.csv is missing entries for: {missing_str}")

    return labels_by_path


def build_trials(data_dir: Path, labels_by_path: dict[Path, ImageLabel]) -> list[Trial]:
    trials: list[Trial] = []
    image_paths: list[Path] = []

    for category in CATEGORY_ORDER:
        category_dir = data_dir / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Missing required folder: {category_dir}")

        category_image_paths = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

        if not category_image_paths:
            raise FileNotFoundError(
                f"No supported image files found in required folder: {category_dir}"
            )

        image_paths.extend(category_image_paths)

    for _ in range(TRIAL_REPETITIONS):
        for image_path in image_paths:
            image_label = labels_by_path[image_path.resolve()]
            trials.append(
                Trial(
                    category=image_label.category,
                    image_id=image_label.image_id,
                    image_path=image_path,
                    rotation_degrees=180 if image_path.stem.endswith(ROTATED_IMAGE_SUFFIX) else 0,
                )
            )

    random.shuffle(trials)

    return trials


def sanitize_participant_id(participant_id: str) -> str:
    normalized = participant_id.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    sanitized = sanitized.strip("._-")
    return sanitized or "participant"


class TrialRunner:
    def __init__(
        self,
        root: tk.Tk,
        trials: list[Trial],
        output_root: Path,
        labels_by_path: dict[Path, ImageLabel],
    ) -> None:
        self.root = root
        self.trials = trials
        self.output_root = output_root
        self.labels_by_path = labels_by_path
        self.image_id_by_category_and_name = {
            (label.category, label.image_name): label.image_id
            for label in labels_by_path.values()
        }
        self.output_path: Path | None = None
        self.output_handle = None
        self.writer: csv.DictWriter | None = None
        self.participant_id = ""
        self.participant_id_safe = ""
        self.current_index = 0
        self.current_photo: ImageTk.PhotoImage | None = None
        self.hide_image_after_id: str | None = None
        self.instructions_active = True
        self.awaiting_batch_resume = False

        if len(self.trials) % BATCH_COUNT != 0:
            raise ValueError(
                f"Total trials ({len(self.trials)}) is not divisible by {BATCH_COUNT}, so equal sets cannot be created."
            )
        self.batch_count = BATCH_COUNT
        self.batch_size = len(self.trials) // self.batch_count

        self.root.title("Top Dent Trial Runner")
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=APP_BG)

        self.status_var = tk.StringVar()
        self.prompt_var = tk.StringVar(value="Which stimulus is convex? (1 = top, 2 = bottom)")
        self.break_message_var = tk.StringVar()
        self.participant_id_var = tk.StringVar()
        self.instructions_text = (
            "For each image, exactly one stimulus is convex and one is concave. If they both appear convex, select the one that bumps out more.\n\n"
            "After viewing each image, indicate which stimulus is convex.\n\n"
            "Press 1 if the TOP stimulus is convex.\n"
            "Press 2 if the BOTTOM stimulus is convex.\n\n"
            "Trials are split into 4 equal sets. After each set, choose Continue or Take Break.\n\n"
            "[SPACE] to continue"
        )

        self.main_frame = tk.Frame(root, bg=APP_BG)
        self.main_frame.pack(expand=True, fill="both", padx=36, pady=30)

        self.header_panel = tk.Frame(
            self.main_frame,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
        )
        self.header_panel.pack(fill="x", pady=(0, 18))

        self.header_label = tk.Label(
            self.header_panel,
            textvariable=self.status_var,
            font=("Helvetica", 18, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
        )
        self.header_label.pack(pady=(18, 6))

        self.prompt_label = tk.Label(
            self.header_panel,
            textvariable=self.prompt_var,
            font=("Helvetica", 13),
            fg=TEXT_MUTED,
            bg=CARD_BG,
        )
        self.prompt_label.pack(pady=(0, 18))

        self.image_panel = tk.Frame(
            self.main_frame,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
        )
        self.image_panel.pack(expand=True, fill="both")

        self.image_label = tk.Label(
            self.image_panel,
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Helvetica", 14),
        )
        self.image_label.pack(expand=True, fill="both", padx=28, pady=28)

        self.screen_frame = tk.Frame(self.main_frame, bg=APP_BG)
        self.screen_frame.pack(pady=(18, 0))

        self.id_screen = tk.Frame(
            self.screen_frame,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
            padx=40,
            pady=36,
        )
        self.instructions_screen = tk.Frame(
            self.screen_frame,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
            padx=40,
            pady=36,
        )
        self.trial_controls = tk.Frame(self.screen_frame, bg=APP_BG)
        self.break_screen = tk.Frame(
            self.screen_frame,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
            bd=0,
            padx=40,
            pady=36,
        )

        self.id_screen_label = tk.Label(
            self.id_screen,
            text="Enter the participant ID to begin.",
            font=("Helvetica", 18, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
            justify="center",
            wraplength=600,
        )
        self.id_screen_label.pack(pady=(0, 20))

        self.id_screen_hint = tk.Label(
            self.id_screen,
            text="This ID will be used to name the results folder and CSV file.",
            font=("Helvetica", 12),
            fg=TEXT_MUTED,
            bg=CARD_BG,
            justify="center",
            wraplength=540,
        )
        self.id_screen_hint.pack(pady=(0, 24))

        self.participant_frame = tk.Frame(self.id_screen, bg=CARD_BG)
        self.participant_frame.pack(fill="x", pady=(0, 28))

        self.participant_label = tk.Label(
            self.participant_frame,
            text="Participant ID:",
            font=("Helvetica", 13, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
            anchor="w",
        )
        self.participant_label.pack(fill="x", pady=(0, 8))

        self.participant_entry = tk.Entry(
            self.participant_frame,
            textvariable=self.participant_id_var,
            font=("Helvetica", 16),
            width=28,
            justify="left",
            bd=0,
            highlightthickness=2,
            highlightbackground=CARD_BORDER,
            highlightcolor=PRIMARY_BUTTON_BG,
            relief="flat",
            bg="#f7faf9",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
        )
        self.participant_entry.pack(fill="x", ipady=12)

        self.id_continue_button = tk.Button(
            self.id_screen,
            text="Continue",
            width=20,
            font=("Helvetica", 13, "bold"),
            command=self.go_to_instructions,
            **self._primary_button_style(),
        )
        self.id_continue_button.pack()

        self.instructions_label = tk.Label(
            self.instructions_screen,
            text=f"{self.instructions_text}",
            font=("Helvetica", 15),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
            justify="center",
            wraplength=620,
        )
        self.instructions_label.pack(pady=(0, 24))

        # self.instructions_keys = tk.Label(
        #     self.instructions_screen,
        #     text="During the task: press C for Convex or V for Concave.",
        #     font=("Helvetica", 12),
        #     fg=TEXT_MUTED,
        #     bg=CARD_BG,
        #     justify="center",
        #     wraplength=620,
        # )
        # self.instructions_keys.pack(pady=(0, 28))

        self.start_button = tk.Button(
            self.instructions_screen,
            text="Start",
            width=20,
            font=("Helvetica", 13, "bold"),
            command=self.start_trials,
            **self._primary_button_style(),
        )
        self.start_button.pack()

        self.break_label = tk.Label(
            self.break_screen,
            textvariable=self.break_message_var,
            font=("Helvetica", 15),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
            justify="center",
            wraplength=620,
        )
        self.break_label.pack(pady=(0, 28))

        self.break_buttons = tk.Frame(self.break_screen, bg=CARD_BG)
        self.break_buttons.pack()

        self.take_break_button = tk.Button(
            self.break_buttons,
            text="Take Break",
            width=18,
            font=("Helvetica", 13, "bold"),
            command=self.take_break,
            **self._primary_button_style(),
        )
        self.take_break_button.pack(side="left", padx=12)

        self.continue_button = tk.Button(
            self.break_buttons,
            text="Continue",
            width=18,
            font=("Helvetica", 13, "bold"),
            command=self.continue_after_break,
            **self._primary_button_style(),
        )
        self.continue_button.pack(side="left", padx=12)

        # self.convex_button = tk.Button(
        #     self.trial_controls,
        #     text="Top (1)",
        #     width=18,
        #     font=("Helvetica", 13, "bold"),
        #     command=lambda: self.record_response("convex"),
        #     **self._response_button_style(
        #         bg=CONVEX_BUTTON_BG,
        #         active_bg=CONVEX_BUTTON_ACTIVE_BG,
        #         fg="#ffffff",
        #     ),
        # )
        # self.convex_button.pack(side="left", padx=12)

        # self.concave_button = tk.Button(
        #     self.trial_controls,
        #     text="Bottom (2)",
        #     width=18,
        #     font=("Helvetica", 13, "bold"),
        #     command=lambda: self.record_response("concave"),
        #     **self._response_button_style(
        #         bg=CONCAVE_BUTTON_BG,
        #         active_bg=CONCAVE_BUTTON_ACTIVE_BG,
        #         fg="#ffffff",
        #     ),
        # )
        # self.concave_button.pack(side="left", padx=12)

        self.root.bind("<Key-1>", lambda _e: self.record_response("top"))
        self.root.bind("<Key-2>", lambda _e: self.record_response("bottom"))
        self.root.bind("<space>", self.handle_return_key)
        self.root.bind("<Return>", self.handle_return_key)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.show_id_screen()
        self.participant_entry.focus_set()

    def _primary_button_style(self) -> dict[str, object]:
        return {
            "bg": PRIMARY_BUTTON_BG,
            "fg": "#ffffff",
            "activebackground": PRIMARY_BUTTON_ACTIVE_BG,
            "activeforeground": "#ffffff",
            "disabledforeground": "#ffffff",
            "relief": "flat",
            "bd": 0,
            "highlightbackground": PRIMARY_BUTTON_BG,
            "highlightcolor": PRIMARY_BUTTON_BG,
            "highlightthickness": 0,
            "overrelief": "flat",
            "takefocus": False,
            "cursor": "hand2",
            "padx": 18,
            "pady": 12,
        }

    def _response_button_style(self, bg: str, active_bg: str, fg: str) -> dict[str, object]:
        return {
            "bg": bg,
            "fg": fg,
            "activebackground": active_bg,
            "activeforeground": fg,
            "disabledforeground": fg,
            "relief": "flat",
            "bd": 0,
            "highlightbackground": bg,
            "highlightcolor": bg,
            "highlightthickness": 0,
            "overrelief": "flat",
            "takefocus": False,
            "cursor": "hand2",
            "padx": 16,
            "pady": 14,
        }

    def _initialize_output(self, participant_id: str) -> None:
        self.participant_id = participant_id.strip()
        self.participant_id_safe = sanitize_participant_id(self.participant_id)
        participant_dir = self.output_root / self.participant_id_safe
        participant_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = participant_dir / f"{self.participant_id_safe}_labels.csv"

        self._upgrade_output_file_if_needed()

        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            self.output_handle = self.output_path.open("a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.output_handle, fieldnames=OUTPUT_FIELDNAMES)
            return

        with self.output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.writer(output_handle)
            writer.writerow(OUTPUT_FIELDNAMES)
        self.output_handle = self.output_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.output_handle, fieldnames=OUTPUT_FIELDNAMES)

    def _lookup_image_id(
        self,
        image_path_value: str | None,
        category: str | None,
        image_name: str | None,
    ) -> int:
        if image_path_value:
            image_label = self.labels_by_path.get(Path(image_path_value).resolve())
            if image_label is not None:
                return image_label.image_id

        if category and image_name:
            image_id = self.image_id_by_category_and_name.get((category, image_name))
            if image_id is not None:
                return image_id

        raise KeyError(f"Unable to determine image_id for category={category}, image_name={image_name}")

    def _upgrade_output_file_if_needed(self) -> None:
        if self.output_path is None or not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return

        with self.output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "image_id" in reader.fieldnames:
                return
            rows = list(reader)

        upgraded_rows: list[dict[str, object]] = []
        for row in rows:
            upgraded_rows.append(
                {
                    "participant_id": row.get("participant_id", ""),
                    "trial_number": row.get("trial_number", ""),
                    "category": row.get("category", ""),
                    "image_id": self._lookup_image_id(
                        row.get("image_path"),
                        row.get("category"),
                        row.get("image_name"),
                    ),
                    "image_name": row.get("image_name", ""),
                    "image_path": row.get("image_path", ""),
                    "rotation_degrees": row.get("rotation_degrees", ""),
                    "response": row.get("response", ""),
                    "timestamp": row.get("timestamp", ""),
                }
            )

        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(upgraded_rows)

    def show_current_trial(self) -> None:
        if self.current_index >= len(self.trials):
            self.finish_experiment()
            return

        self._cancel_pending_image_hide()
        trial = self.trials[self.current_index]
        batch_number = (self.current_index // self.batch_size) + 1
        self.status_var.set(
            f"Set {batch_number} of {self.batch_count} - Trial {self.current_index + 1} of {len(self.trials)}"
        )

        image = Image.open(trial.image_path)
        image = ImageOps.exif_transpose(image)

        image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo, text="")
        self.hide_image_after_id = self.root.after(IMAGE_DISPLAY_MS, self.hide_current_image)

    def _cancel_pending_image_hide(self) -> None:
        if self.hide_image_after_id is not None:
            self.root.after_cancel(self.hide_image_after_id)
            self.hide_image_after_id = None

    def hide_current_image(self) -> None:
        self.hide_image_after_id = None
        self.current_photo = None
        self.image_label.configure(image="", text="")

    def _show_screen(self, screen: tk.Frame | None) -> None:
        for frame in (self.id_screen, self.instructions_screen, self.trial_controls, self.break_screen):
            frame.pack_forget()

        if screen is not None:
            screen.pack()

    def show_break_screen(self) -> None:
        completed_batches = self.current_index // self.batch_size
        next_batch = completed_batches + 1
        self.awaiting_batch_resume = True
        self.status_var.set(f"Set {completed_batches} of {self.batch_count} complete")
        self.prompt_var.set("Take a break or continue.")
        self.current_photo = None
        self.image_label.configure(image="", text="")
        self.break_message_var.set(
            f"Set {completed_batches} of {self.batch_count} complete.\n\n"
            f"Choose Continue to start set {next_batch}, or Take Break."
        )
        self.take_break_button.configure(state="normal")
        self.continue_button.configure(state="normal")
        self._show_screen(self.break_screen)

    def take_break(self) -> None:
        if not self.awaiting_batch_resume:
            return

        completed_batches = self.current_index // self.batch_size
        next_batch = completed_batches + 1
        self.status_var.set(f"Break after set {completed_batches} of {self.batch_count}")
        self.prompt_var.set("Break in progress. Press Continue when ready.")
        self.break_message_var.set(
            f"Break started.\n\nPress Continue when you are ready for set {next_batch} of {self.batch_count}."
        )
        self.take_break_button.configure(state="disabled")

    def continue_after_break(self) -> None:
        if not self.awaiting_batch_resume:
            return

        self.awaiting_batch_resume = False
        self._show_screen(self.trial_controls)
        self.prompt_var.set("Indicate which stimulus is convex. Press 1 for top, press 2 for bottom.")
        self.show_current_trial()

    def show_id_screen(self) -> None:
        self.awaiting_batch_resume = False
        self.status_var.set("Participant Setup")
        self.prompt_var.set("Enter a participant ID to continue.")
        self.image_label.configure(
            image="",
            text="The trial image will appear here once the task begins.",
            justify="center",
        )
        self._show_screen(self.id_screen)
        # self.convex_button.configure(state="disabled")
        # self.concave_button.configure(state="disabled")

    def go_to_instructions(self) -> None:
        participant_id = self.participant_id_var.get().strip()
        if not participant_id:
            messagebox.showerror("Participant ID required", "Enter a participant ID before continuing.")
            self.participant_entry.focus_set()
            return

        self.status_var.set("Instructions")
        self.prompt_var.set("Read the instructions, then press Start.")
        self.image_label.configure(
            image="",
            text="Review the task instructions below before starting.",
            justify="center",
        )
        self._show_screen(self.instructions_screen)
        self.start_button.configure(state="normal")

    def start_trials(self) -> None:
        if not self.instructions_active:
            return

        participant_id = self.participant_id_var.get().strip()
        self._initialize_output(participant_id)

        self.instructions_active = False
        self.awaiting_batch_resume = False
        self._show_screen(self.trial_controls)
        # self.convex_button.configure(state="normal")
        # self.concave_button.configure(state="normal")
        self.prompt_var.set(
            "Indicate which stimulus is convex. Press 1 for top, press 2 for bottom."
        )
        self.show_current_trial()

    def handle_return_key(self, _event: tk.Event) -> None:
        if self.instructions_active and self.id_screen.winfo_manager():
            self.go_to_instructions()
            return
        if self.instructions_active and self.instructions_screen.winfo_manager():
            self.start_trials()
            return
        if self.awaiting_batch_resume:
            self.continue_after_break()

    def record_response(self, response: str) -> None:
        if self.instructions_active or self.awaiting_batch_resume or self.current_index >= len(self.trials):
            return

        trial = self.trials[self.current_index]
        if self.writer is None or self.output_handle is None:
            raise RuntimeError("Output file is not initialized.")

        self.writer.writerow(
            {
                "participant_id": self.participant_id,
                "trial_number": self.current_index + 1,
                "category": trial.category,
                "image_id": trial.image_id,
                "image_name": trial.image_path.name,
                "image_path": str(trial.image_path),
                "rotation_degrees": trial.rotation_degrees,
                "response": response,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.output_handle.flush()

        self._cancel_pending_image_hide()
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self.show_current_trial()
            return
        if self.current_index % self.batch_size == 0:
            self.show_break_screen()
            return
        self.show_current_trial()

    def finish_experiment(self) -> None:
        self._cancel_pending_image_hide()
        self.awaiting_batch_resume = False
        self.image_label.configure(image="", text="")
        self.status_var.set("Experiment complete")
        saved_name = self.output_path.name if self.output_path is not None else "results file"
        self.prompt_var.set(f"Responses saved to {saved_name}")
        # self.convex_button.configure(state="disabled")
        # self.concave_button.configure(state="disabled")
        self.image_label.configure(
            image="",
            text="All trials are complete.",
            justify="center",
        )
        if self.output_path is not None:
            messagebox.showinfo("Complete", f"All trials finished.\nSaved responses to {self.output_path}")

    def on_close(self) -> None:
        self._cancel_pending_image_hide()
        if self.output_handle is not None and not self.output_handle.closed:
            self.output_handle.close()
        self.root.destroy()


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "stimuli"
    output_root = project_dir / OUTPUT_DIRNAME
    labels_path = project_dir / LABELS_FILENAME

    try:
        labels_by_path = load_image_labels(labels_path, data_dir)
        trials = build_trials(data_dir, labels_by_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    root = tk.Tk()
    try:
        runner = TrialRunner(
            root,
            trials=trials,
            output_root=output_root,
            labels_by_path=labels_by_path,
        )
    except ValueError as exc:
        root.destroy()
        print(exc, file=sys.stderr)
        return 1
    try:
        root.mainloop()
    finally:
        if runner.output_handle is not None and not runner.output_handle.closed:
            runner.output_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
