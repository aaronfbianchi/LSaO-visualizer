import tkinter as tk
import traceback
import typing
import webbrowser
from tkinter import ttk

import numpy as np
from progress.bar import IncrementalBar as Bar

import gui
from backend.common import Visualizer, VizParam
from util import VideoFrame, read_audio_samples, spawn_ffmpeg

from . import preview, widgets


class SettingsWindow:
    CURRENT: typing.ClassVar[SettingsWindow]

    params: dict[str, tuple[type, VizParam]]
    tk_vars: dict[str, tk.Variable]
    input_path_var: tk.StringVar
    output_path_var: tk.StringVar
    visualizer: Visualizer

    def __init__(self, master: tk.Toplevel, visualizer: type[Visualizer]) -> None:
        gui.root.withdraw()
        self.master = master
        self.master.title(visualizer.title)
        self.params = visualizer.gather_params()
        self.tk_vars = {}
        self.visualizer = visualizer()
        self.build_ui()

        SettingsWindow.CURRENT = self

    def build_ui(self) -> None:
        row_num = 0

        widgets.back_button(gui.root, self.master)
        ignored_label = tk.Label(self.master, text="WARNING: Only parameters with the * mark are taken into account for the live preview!", anchor="w", justify="left")
        ignored_label.grid(row=row_num, column=0, columnspan = 3, padx=80, pady=10, sticky="w")
        row_num += 1

        self.input_path_var = tk.StringVar()
        self.input_path_var.trace_add("write", lambda *_: self.validate_params(True))
        widgets.file_input(self.master, "Input audio", row_num, self.input_path_var)
        row_num += 1

        self.output_path_var = tk.StringVar(value="output.mp4")
        self.output_path_var.trace_add("write", lambda *_: self.validate_params(True))
        widgets.file_output(self.master, "Output video", row_num, self.output_path_var)
        row_num += 1

        for param_name, (param_type, metadata) in self.params.items():
            var: tk.Variable
            var2: tk.Variable | None = None
            initial_value = getattr(self.visualizer, param_name)

            label = metadata.name
            if not metadata.ignored_in_live_mode:
                label = "* " + label

            if isinstance(param_type, typing.TypeAliasType):
                param_type = param_type.__value__

            if param_type is int:
                var = tk.IntVar(value=initial_value)
                if metadata.values:
                    widgets.combobox(self.master, label, var, row_num, metadata.values, metadata.description)
                else:
                    widgets.number(self.master, label, var, row_num, metadata.description or "")
            elif param_type is bool:
                var = tk.BooleanVar(value=initial_value)
                widgets.toggle(self.master, label, var, row_num, metadata.description or "")
            elif param_type is float:
                var = tk.DoubleVar(value=initial_value)
                if metadata.values:
                    widgets.combobox(self.master, label, var, row_num, metadata.values, metadata.description)
                else:
                    widgets.number(self.master, label, var, row_num, metadata.description or "")
            elif param_type is str:
                var = tk.StringVar(value=initial_value)
                widgets.textbox(self.master, label, var, row_num, metadata.description or "")
            elif param_type == tuple[int, int]:
                var = tk.IntVar(value=initial_value[0])
                var2 = tk.IntVar(value=initial_value[1])
                widgets.dual_combobox(self.master, label, var, metadata.divider, var2, row_num, metadata.values or [], metadata.values2 or [], metadata.description)
            elif param_type == tuple[float, float]:
                var = tk.DoubleVar(value=initial_value[0])
                var2 = tk.DoubleVar(value=initial_value[1])
                widgets.dual_combobox(self.master, label, var, metadata.divider, var2, row_num, metadata.values or [], metadata.values2 or [], metadata.description)
            elif typing.get_origin(param_type) is typing.Literal:
                # Enum-style types
                values = typing.get_args(param_type)
                assert isinstance(values[0], str)
                var = tk.StringVar(value=initial_value)
                widgets.combobox(self.master, label, var, row_num, values, metadata.description, readonly=True)
            else:
                print(f"unhandled type: {param_type} (metatype is {type(param_type)}) (origin is {typing.get_origin(param_type)})")
                continue

            var.trace_add("write", lambda *_: self.validate_params(True))

            self.tk_vars[param_name] = var
            if var2 is not None:
                self.tk_vars[param_name + "2"] = var2
                var2.trace_add("write", lambda *_: self.validate_params(True))

            row_num += 1

        self.action_button = tk.Button(self.master, text="Render video", command=self.generate_video)
        self.action_button.grid(row=row_num, column=1, pady=10)

        self.loading_label = tk.Label(self.master, font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

        self.progress_bar = ttk.Progressbar(self.master, length=300)
        self.progress_bar.grid(row=row_num, column=2, padx=10, pady=5, sticky='e')
        self.progress_bar.grid_remove()

        widgets.preview_toggle(self.master, row_num, preview.show_window)

    def validate_params(self, update_ui: bool) -> bool:
        err_msg = None
        enable_button = True
        for param_name, (_ty, metadata) in self.params.items():
            try:
                val = self.tk_vars[param_name].get()
                if param_name + "2" in self.tk_vars:
                    val2 = self.tk_vars[param_name + "2"].get()
                    val = (val, val2)
                metadata.validator(val)
            except Exception as e:
                err_msg = f"{metadata.name}: {e.args[0]}"
                enable_button = False
                break

        enable_button = enable_button and bool(self.input_path_var.get()) and bool(self.output_path_var.get())
        self.action_button.config(state=tk.NORMAL if enable_button else tk.DISABLED)

        if update_ui:
            if err_msg is None:
                self.loading_label.grid_remove()
            else:
                self.loading_label.config(fg="Red", text=err_msg)
                self.loading_label.grid()

        return err_msg is None

    def update_params(self) -> None:
        if not self.validate_params(False):
            return

        for param_name, tk_var in self.tk_vars.items():
            try:
                val = tk_var.get()
                if param_name + "2" in self.tk_vars:
                    val2 = self.tk_vars[param_name + "2"].get()
                    val = (val, val2)
            except tk.TclError:
                continue

            setattr(self.visualizer, param_name, val)

    def generate_video(self) -> None:
        self.update_params()
        self.loading_label.grid()
        self.loading_label.config(text="Loading...", fg="blue")

        try:
            input_audio, sample_rate = read_audio_samples(self.input_path_var.get())
            audio = self.visualizer.prepare_audio(input_audio)

            frame_sequence = self.visualizer.generate_video(audio, sample_rate)

            ffmpeg_process = spawn_ffmpeg(
                resolution=self.visualizer.resolution,
                framerate=self.visualizer.framerate,
                crf=self.visualizer.compression,
                input_audio_path=self.input_path_var.get(),
                output_video_path=self.output_path_var.get(),
            )
            assert ffmpeg_process.stdin is not None

            self.progress_bar.grid()

            num_frames = len(frame_sequence)
            for i, frame_data in enumerate(Bar('Rendering').iter(frame_sequence)):
                # Need to pack along the horizontal axis because ffmpeg expects each line to be a whole number of bytes
                ffmpeg_process.stdin.write(np.packbits(frame_data, axis=1))
                self.loading_label.config(fg="Blue", text=f"{i+1}/{num_frames}")
                self.loading_label.grid()
                self.progress_bar['value'] = 100 * (i / num_frames)
                self.master.update()

            self.progress_bar.grid_remove()

            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            self.update_loading_label(i, num_frames, True, "Done, my dood!")

        except Exception as exc:  # noqa: BLE001
            traceback.print_exception(exc)
            message = "".join(traceback.format_exception_only(exc)).strip()
            self.loading_label.config(fg="Red", text=message)
            self.master.update()

    def live_preview(self, block: np.ndarray[tuple[int, int], np.dtype[np.float32]], resolution: tuple[int, int]) -> VideoFrame:
        self.update_params()
        return self.visualizer.live_preview(self.visualizer.prepare_audio(block), resolution)

    def update_loading_label(self, progress: int, total: int, text_state: bool, text_message: str) -> None:
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class HelpWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Help!!!!!!!!!!!!!111ONE")

        row_num = 0
        warning_label = tk.Label(self.master, text="HELP iM TRAPPED IN A VISUALIZER FACTORY IN  [R E D A C T E D]", fg_color="red")
        warning_label.grid(row=row_num, column=0, columnspan=3, padx=(5, 5), pady=(5, 0), sticky="we")

        row_num += 1
        self.action_button = tk.Button(self.master, text="Donate for rescue", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)

        row_num += 1
        warning_label = tk.Label(self.master, text="Actually no. This program is totally free. You don't have to pay anything.\n But if you feel the need, donations are appreciated :)", fg_color="black")
        warning_label.grid(row=row_num, column=0, columnspan=3, padx=(5, 5), pady=(5, 0), sticky="we")

    def perform_action(self):
        webbrowser.open("https://aaron-f-bianchi.itch.io/lsao/purchase")
