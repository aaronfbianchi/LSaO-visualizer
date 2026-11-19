from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
import webbrowser
import ctypes.util

from hidpi_tk import DPIAwareTk
from PIL import Image, ImageTk

from backend import line_plot, scatter_plot, threshold
from backend.common import Visualizer

from . import preview
from .windows import SettingsWindow

lsao_version = "v2.99"

root: tk.Tk

def handle_child_close(child: tk.Toplevel, root: tk.Tk, exit_if_no_root: bool = True) -> None:
    """
    Attach a WM_DELETE_WINDOW handler to a child window.

    If exit_if_no_root is True, closing the child will also destroy the root.
    Otherwise, it will deiconify the root if it's hidden.
    """
    def on_close() -> None:
        child.destroy()
        if exit_if_no_root or not root.winfo_viewable():
            root.destroy()  # exit the program
        else:
            root.deiconify()  # show hidden root

    child.protocol("WM_DELETE_WINDOW", on_close)

def option_logo() -> None:
    webbrowser.open("https://aaron-f-bianchi.itch.io/lsao/purchase")

#def optionhelp():
#    help_window = tk.Toplevel(root)
#    HelpWindow(help_window)

class GifButton(tk.Button):
    gif_playing: bool
    current_frame: int

def update_gif(button: GifButton, frames: list[ImageTk.PhotoImage]) -> None:
    if button.gif_playing:
        frame_index = button.current_frame
        try:
            gif_frame = frames[frame_index]
        except IndexError:
            frame_index = 0
            gif_frame = frames[frame_index]

        button.config(image=gif_frame)
        button.current_frame = frame_index + 1
        button.after(20, update_gif, button, frames)

def start_gif(_event: tk.Event[GifButton], button: GifButton, frames: list[ImageTk.PhotoImage]) -> None:
    if not button.gif_playing:
        button.gif_playing = True
        update_gif(button, frames)

def stop_gif(_event: tk.Event[GifButton], button: GifButton) -> None:
    button.gif_playing = False

PACKAGED: bool = getattr(sys, 'frozen', False)
def resource(filename: str) -> str:
    path = os.path.join("resources", filename)
    if PACKAGED and os.name == "nt":
        path = os.path.join(sys._MEIPASS, path) # type: ignore
    return path

def load_gif(filename: str) -> list[ImageTk.PhotoImage]:
    gif = Image.open(resource(filename))
    frames = []
    try:
        while True:
            frame = ImageTk.PhotoImage(gif.copy())
            frames.append(frame)
            gif.seek(len(frames))
    except EOFError:
        pass
    return frames

button_width: int = 1
button_height: int = 1

def make_button(
    text: str,
    visualizer: type[Visualizer],
    gif: list[ImageTk.PhotoImage],
    frame: int,
    row_num: int,
    col_num: int,
) -> GifButton:
    def show() -> None:
        window = tk.Toplevel(root)
        window.resizable(False, False)
        SettingsWindow(window, visualizer)
        handle_child_close(window, root)

    button = GifButton(root, image=gif[frame], text=text, compound=tk.BOTTOM, command=show, width=button_width, height=button_height)
    button.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
    button.gif_playing = False
    button.current_frame = frame
    button.bind("<Enter>", lambda e: start_gif(e, button, gif))
    button.bind("<Leave>", lambda e: stop_gif(e, button))
    return button

credit_counter: int = 0
def make_credits_window():
    global credit_counter
    credit_counter += 1

    credits_window = tk.Toplevel(root)
    credits_window.title("Credits")
    # credits_window.geometry("600x450")
    credits_window.resizable(False, False)

    credits_title = tk.Label(credits_window, text="LSaO Visualizer " + lsao_version, font=("Helvetica", 15, "bold"))
    credits_title.grid(row=0,column=0,padx=20,pady=20, sticky="w")

    def go_back():
        root.deiconify() # VUELVE A MOSTRAR LA VENTANA PRINCIPAL
        for w in root.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.destroy()

    credits_body = tk.Label(credits_window,
        text=("This program is free software, licensed under GPLv3!\n\n" +
lsao_version + """ has also been built with the contributions of:"""),
        font=("Helvetica", 10),
        anchor='w',
        justify='left')
    credits_body.grid(row=1,column=0,padx=20,pady=0, sticky="w")

    link_The0x539 = tk.Label(credits_window, text="The0x539", font=("Helvetica", 10, "underline"), fg="blue", justify='left')
    link_The0x539.grid(row=2, column=0, columnspan=3, padx=20, pady=(15,5), sticky="w")
    link_The0x539.bind("<Button-1>", lambda e: webbrowser.open("https://pub.colonq.computer/~the0x539"))

    credits_The0x539 = tk.Label(credits_window,
    text=( """  - Major code cleanup.
  - Optimizations in Waveforms and Envelope modes.
  - Huge optimization in RAM usage for scatter plots.
  - Do prettier scatter plot interpolation.
  - Code an in-house spline interpolation function.
  - Dynamic GUI generation.
  - CLI support.
  - Good practices.

Huge thanks!!!"""),
        font=("Helvetica", 10),
        anchor='w',
        justify='left')
    credits_The0x539.grid(row=3,column=0,padx=20,pady=0, sticky="w")

    counter = tk.Label(credits_window, text="Times you have opened this window: " + str(credit_counter), anchor="w", justify="left")
    counter.grid(row=4,column=0,padx=20,pady=(10,0), sticky="w")

    go_back_credits = tk.Button(credits_window, text="NO GO BACK GO BACK!!!", command=go_back)
    go_back_credits.grid(row=5,column=0,padx=20,pady=20, sticky="w")

def gui_main() -> None:
    global root
    root = DPIAwareTk()
    root.title("LSaO Visualizer "+lsao_version)
    root.resizable(False, False)

    icon = tk.PhotoImage(file=resource("lsao_icon.png"))
    root.iconphoto(True, icon)

    # RECEIVING THE AUDIO
    threading.Thread(target=preview.audio_thread, daemon=True).start()

    logo = tk.PhotoImage(file=resource("lsao_banner.png"))

    row_num = 0

    # LOGO
    button_logo = tk.Label(root, image=logo, text="", justify='center')
    button_logo.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="we")
    button_logo.bind("<Button-1>", lambda e: webbrowser.open("https://aaron-f-bianchi.itch.io/lsao"))


    row_num += 1
    # INITIAL TEXT
    initial_text = "Things to know:\n"
    global button_width, button_height
    if os.name == 'nt':
        button_width = 100
        button_height = 104
        print("Running on Windows")
        initial_text = initial_text + "\n- The default Windows video player isn't going to play the generated\n  videos correctly. Try a better video player (VLC, for example).\n- This program uses FFmpeg."
    elif os.name == 'posix':
        button_width = 90
        button_height = 110
        print("Running on Linux")
        initial_text = initial_text + "\n- You have to install FFmpeg and libportaudio2 if not\n  already installed."
        lib_path = ctypes.util.find_library("portaudio")
        if lib_path:
            initial_text = ""
        else:
            initial_label = tk.Label(root, text=initial_text, font=("Helvetica", 10), anchor='w', justify='left')
            initial_label.grid(row=row_num, column=0, columnspan=4, padx=5, pady=10, sticky="w")


    row_num += 1
    col_num = 0

    modes: list[tuple[str, type[Visualizer], str, int]] = [
        ("Linear Spec.", line_plot.SpectrumVisualizer, "resized_img_spec.gif", 38),
        ("Lin. Spec. (dB)", line_plot.SpectrumdBVisualizer, "resized_img_specdB.gif", 24),
        ("Spec. Balance", line_plot.SpecBalanceVisualizer, "resized_img_spec_balance.gif", 50),
        ("Spectrogram", threshold.SpectrogramVisualizer, "resized_img_spectrogram.gif", 51),
        ("Histogram", line_plot.HistogramVisualizer, "resized_img_histogram.gif", 5),
        ("Short Wave", line_plot.WaveformVisualizer, "resized_img_swav.gif", 53),
        ("Long Wave", line_plot.LongWaveformVisualizer, "resized_img_lwav.gif", 30),
        ("Recurrence", threshold.RecurrenceVisualizer, "resized_img_recurrence.gif", 111),
        ("Oscilloscope", scatter_plot.OscilloscopeVisualizer, "resized_img_osc.gif", 109),
        ("Vectorscope", scatter_plot.VectorscopeVisualizer, "resized_img_vectorscope.gif", 159),
        ("Polar (Mono)", scatter_plot.PolarMonoVisualizer, "resized_img_polar.gif", 228),
        ("Polar (Stereo)", scatter_plot.PolarStereoVisualizer, "resized_img_polar_stereo.gif", 575),
        ("Poincaré", scatter_plot.PoincareVisualizer, "resized_img_poincare.gif", 41),
        ("Delay Embed", scatter_plot.DelayEmbedVisualizer, "resized_img_embed2.gif", 97),
        ("False Chladni", threshold.FalseChladniVisualizer, "resized_img_false_chladni.gif", 11),
        ("Envelope", line_plot.EnvelopeVisualizer, "resized_img_envelope.gif", 11),
        ("Chladni", threshold.ChladniVisualizer, "resized_img_chladni.gif", 92),
    ]

    for (label, visualizer, gif_path, frame_number) in modes:
        gif = load_gif(gif_path)
        make_button(label, visualizer, gif, frame_number, row_num, col_num)
        col_num += 1
        if col_num == 4:
            col_num = 0
            row_num += 1

    row_num += 1
    col_num = 0
    credits_button = tk.Button(root, text="About", command=make_credits_window)
    credits_button.grid(row=row_num, column=col_num, columnspan=2, padx=0, pady=0, sticky="we")

    col_num += 2
    credits_label = tk.Label(root, text="© 2026 Aaron F. Bianchi", font=("Helvetica", 10), fg="blue", justify='center')
    credits_label.grid(row=row_num, column=col_num, columnspan=3, pady=5, sticky="ew")
    credits_label.bind("<Button-1>", lambda e: webbrowser.open("https://aaronfbianchi.github.io/"))

    root.mainloop()
