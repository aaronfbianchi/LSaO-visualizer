import os
import queue
import threading
import time
import tkinter as tk
import warnings

import numpy as np
from PIL import Image, ImageTk

from .windows import SettingsWindow

if os.name == 'posix':
    import sounddevice as sd
elif os.name == 'nt':
    import soundcard as sc


def show_window(self) -> None:
    if hasattr(self, "preview_window") and self.preview_window.winfo_exists():
        self.preview_window.lift()
        return

    self.preview_window = tk.Toplevel(self)
    w = self.preview_window
    w.title("Live Preview")
    w.geometry("400x400")
    w.resizable(True, True)

    start_frame_loop(w)


audio_queue = queue.Queue(maxsize=8)

SAMPLERATE = 48000
BLOCK_SIZE = SAMPLERATE // 60
CHANNELS = 2

def audio_block_stream():
    if os.name == "nt":
        warnings.simplefilter('ignore', sc.SoundcardRuntimeWarning)

        mic = sc.get_microphone(
            sc.default_speaker().name,
            include_loopback=True
        )

        def capture_loop():
            with mic.recorder(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
            ) as rec:
                while True:
                    data = rec.record(BLOCK_SIZE)
                    if not audio_queue.full():
                        audio_queue.put(data.astype(np.float32))

        threading.Thread(target=capture_loop, daemon=True).start()

    else:
        def _callback(indata, frames, time, status):
            if not audio_queue.full():
                audio_queue.put(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=_callback,
        )
        stream.start()

    while True:
        yield audio_queue.get()

latest_block: np.ndarray[tuple[int, int], np.dtype[np.floating]]

def audio_thread():
    global latest_block
    for block in audio_block_stream():
        latest_block = block

def start_frame_loop(window):
    last_time = time.time()

    label = tk.Label(window)  # create a label inside this new window
    label.pack(fill="both", expand=True)

    def update():
        nonlocal last_time
        now = time.time()
        if now - last_time >= 1/60:  # ~60 FPS
            h, w = label.winfo_height(), label.winfo_width()
            if h < 5 or w < 5:
                window.after(1, update)
                return

            block = latest_block
            if block is not None:
                frame = SettingsWindow.CURRENT.live_preview(block, (w, h))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                label.config(image=img)
                label.image = img # type: ignore

            last_time = now

        window.after(1, update)

    update()
