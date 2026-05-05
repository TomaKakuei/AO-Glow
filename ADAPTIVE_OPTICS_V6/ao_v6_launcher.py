from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import ttk


ROOT = Path(__file__).resolve().parent
CONDA_EXE = Path.home() / "anaconda3" / "Scripts" / "conda.exe"
WORKER_PATH = ROOT / "ao_v6_worker.py"
TEMP_DIR = ROOT / "artifacts" / "ui_temp"
TORCH_FALLBACK_DIR = ROOT.parent / "dist" / "AdaptiveOpticsGUI" / "_internal"
ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]


class AdaptiveOpticsV6App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Adaptive Optics V6 Launcher")
        self.root.geometry("1180x840")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.last_run_dir = ""

        self.status_var = tk.StringVar(value="Initializing environment...")
        self.env_var = tk.StringVar(value="Torch: checking...")
        self.output_name_var = tk.StringVar(value="ui_case")
        self.overall_variation_var = tk.StringVar(value="1.0")
        self.shwfs_noise_profile_var = tk.StringVar(value="realistic")

        self.anchor_entries: dict[int, dict[str, tk.Entry]] = {}

        self._build_ui()
        self.root.after(120, self._poll_events)
        self._run_worker("env", None, "environment initialization")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text="Adaptive Optics V6", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.status_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.env_var, foreground="#1d6f42").pack(anchor="w", pady=(2, 10))

        config_frame = ttk.LabelFrame(outer, text="Input Configuration", padding=10)
        config_frame.pack(fill="x")

        row1 = ttk.Frame(config_frame)
        row1.pack(fill="x", pady=(0, 8))
        ttk.Label(row1, text="Output Name").pack(side="left")
        ttk.Entry(row1, textvariable=self.output_name_var, width=26).pack(side="left", padx=(8, 18))
        ttk.Label(row1, text="Overall Variation (mm)").pack(side="left")
        ttk.Entry(row1, textvariable=self.overall_variation_var, width=12).pack(side="left", padx=(8, 0))
        ttk.Label(row1, text="SHWFS Noise Profile").pack(side="left", padx=(18, 0))
        noise_combo = ttk.Combobox(
            row1,
            textvariable=self.shwfs_noise_profile_var,
            values=("none", "minimum", "realistic", "enhanced"),
            width=12,
            state="readonly",
        )
        noise_combo.pack(side="left", padx=(8, 0))

        ttk.Label(
            config_frame,
            text="Per-anchor override fields are requested dx / dy / dz in mm. V6 keeps the hidden-plant truth boundary and adds robust SHWFS image preprocessing, weighted centroiding, and regularized modal fitting on top of the 13x13 noisy sensor model.",
        ).pack(anchor="w", pady=(0, 8))

        table = ttk.Frame(config_frame)
        table.pack(fill="x")
        headers = ["Anchor", "dx_mm", "dy_mm", "dz_mm"]
        for col, title in enumerate(headers):
            ttk.Label(table, text=title, font=("Segoe UI", 9, "bold")).grid(row=0, column=col, padx=6, pady=4, sticky="w")
        for row_index, anchor_surface_id in enumerate(ACTUATOR_IDS, start=1):
            ttk.Label(table, text=str(anchor_surface_id)).grid(row=row_index, column=0, padx=6, pady=3, sticky="w")
            self.anchor_entries[anchor_surface_id] = {}
            for col_index, axis_name in enumerate(("dx_mm", "dy_mm", "dz_mm"), start=1):
                entry = ttk.Entry(table, width=10)
                entry.grid(row=row_index, column=col_index, padx=6, pady=3, sticky="w")
                self.anchor_entries[anchor_surface_id][axis_name] = entry

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=12)
        self.before_button = ttk.Button(button_row, text="Generate Before Calibration", command=self._on_generate_before)
        self.before_button.pack(side="left")
        self.repair_button = ttk.Button(button_row, text="Repair!", command=self._on_repair)
        self.repair_button.pack(side="left", padx=(10, 0))
        self.open_button = ttk.Button(button_row, text="Open Last Run Folder", command=self._open_last_run)
        self.open_button.pack(side="left", padx=(10, 0))

        self.progress = ttk.Progressbar(button_row, mode="indeterminate", length=220)
        self.progress.pack(side="right")

        log_frame = ttk.LabelFrame(outer, text="Run Log", padding=10)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=16)
        self.log_text.pack(fill="both", expand=True)

    def _append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _collect_config_payload(self) -> dict[str, object]:
        output_name = self.output_name_var.get().strip() or "ui_case"
        overall_variation_mm = float(self.overall_variation_var.get().strip() or "1.0")
        overrides: dict[int, dict[str, float]] = {}
        for anchor_surface_id, entry_map in self.anchor_entries.items():
            row_values: dict[str, float] = {}
            for axis_name, entry in entry_map.items():
                raw = entry.get().strip()
                if raw:
                    row_values[axis_name] = float(raw)
            if row_values:
                overrides[int(anchor_surface_id)] = row_values
        return {
            "overall_variation_mm": overall_variation_mm,
            "per_anchor_overrides": overrides,
            "output_name": output_name,
            "shwfs_noise_profile": self.shwfs_noise_profile_var.get().strip() or "realistic",
        }

    def _set_busy(self, is_busy: bool, message: str) -> None:
        self.busy = is_busy
        self.status_var.set(message)
        state = "disabled" if is_busy else "normal"
        self.before_button.configure(state=state)
        self.repair_button.configure(state=state)
        if is_busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _worker_command(self, mode: str, payload: dict[str, object] | None) -> list[str]:
        command = [str(CONDA_EXE), "run", "-n", "ao311", "python", "-u", str(WORKER_PATH), mode]
        if payload is not None:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            temp_path = TEMP_DIR / f"worker_{uuid.uuid4().hex}.json"
            temp_path.write_text(json.dumps(payload), encoding="utf-8")
            command.append("@" + str(temp_path))
        return command

    def _run_worker(self, mode: str, payload: dict[str, object] | None, description: str) -> None:
        if self.busy:
            return
        self._set_busy(True, f"Running {description}...")

        def worker() -> None:
            command = self._worker_command(mode, payload)
            try:
                env = os.environ.copy()
                if TORCH_FALLBACK_DIR.exists():
                    env["PATH"] = str(TORCH_FALLBACK_DIR) + os.pathsep + env.get("PATH", "")
                proc = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as exc:
                self.events.put(("error", f"Failed to start worker: {type(exc).__name__}: {exc}"))
                return

            result_payload: dict[str, object] | None = None
            if proc.stdout is not None:
                for line in proc.stdout:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        self.events.put(("log", stripped))
                        continue
                    event_type = event.get("type")
                    if event_type == "log":
                        self.events.put(("log", str(event.get("message", ""))))
                    elif event_type == "result":
                        result_payload = event.get("payload", {})
                    elif event_type == "error":
                        self.events.put(("error", str(event.get("message", "Unknown worker error."))))
            return_code = proc.wait()
            if return_code != 0 and result_payload is None:
                self.events.put(("error", f"Worker exited with code {return_code}."))
                return
            if result_payload is not None:
                self.events.put(("done", {"mode": mode, **result_payload}))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_before(self) -> None:
        self._run_worker("before", self._collect_config_payload(), "before calibration generation")

    def _on_repair(self) -> None:
        self._run_worker("repair", self._collect_config_payload(), "repair optimization")

    def _open_last_run(self) -> None:
        if not self.last_run_dir:
            self._append_log("No run folder available yet.")
            return
        try:
            os.startfile(self.last_run_dir)  # type: ignore[attr-defined]
        except Exception as exc:
            self._append_log(f"Open folder failed: {type(exc).__name__}: {exc}")

    def _handle_done(self, payload: dict[str, object]) -> None:
        mode = str(payload.get("mode", ""))
        if mode == "env":
            if bool(payload.get("ok")):
                self.env_var.set(
                    f"Torch OK | version {payload.get('torch_version', '')} | CUDA available: {'Yes' if payload.get('torch_cuda_available') else 'No'}"
                )
                self._append_log(f"Python: {payload.get('python_executable', '')}")
                self._append_log(f"Torch version: {payload.get('torch_version', '')}")
                self._set_busy(False, "Environment ready.")
            else:
                self.env_var.set(f"Torch import failed: {payload.get('torch_error', '')}")
                self._append_log(f"Python: {payload.get('python_executable', '')}")
                self._append_log(f"Torch import failed: {payload.get('torch_error', '')}")
                self._set_busy(False, "Environment check failed.")
            return

        self.last_run_dir = str(payload.get("run_dir", ""))
        self._append_log("")
        self._append_log(str(payload.get("summary_text", "")))
        self._append_log(f"Run directory: {self.last_run_dir}")
        if mode == "before":
            self._append_log(f"Before summary: {payload.get('summary_path', '')}")
            self._append_log(f"Before figure: {payload.get('figure_path', '')}")
            self._set_busy(False, "Before calibration outputs are ready.")
        else:
            self._append_log(f"Before summary: {payload.get('before_summary_path', '')}")
            self._append_log(f"Repair summary: {payload.get('repair_summary_path', '')}")
            self._append_log(f"Compare figure: {payload.get('repair_compare_figure_path', '')}")
            self._append_log(f"Metrics figure: {payload.get('repair_metrics_figure_path', '')}")
            self._append_log(f"FOV figure: {payload.get('repair_fov_figure_path', '')}")
            self._set_busy(False, "Repair outputs are ready.")

    def _poll_events(self) -> None:
        while True:
            try:
                event_type, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event_type == "log":
                self._append_log(str(payload))
            elif event_type == "done":
                self._handle_done(payload)  # type: ignore[arg-type]
            elif event_type == "error":
                self._append_log(str(payload))
                self._set_busy(False, "Task failed.")
        self.root.after(120, self._poll_events)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    AdaptiveOpticsV6App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
