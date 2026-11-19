import tkinter as tk
import typing
from tkinter import filedialog, ttk
import shutil
import os
import subprocess


def validate_numeric(value):
    if value == '' or value == '-':
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False

def number(master: tk.Toplevel, label_text: str, variable: tk.IntVar| tk.DoubleVar, row: int, tip: str) -> None:
    tk.Label(master, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(master, textvariable=variable, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    entry.grid(row=row, column=1, padx=10, pady=5, sticky="we")
    entry.config(width=10)

    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def textbox(master: tk.Toplevel, label_text: str, variable: tk.StringVar, row: int, tip: str):
    tk.Label(master, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(master, textvariable=variable, validate="key")
    entry.grid(row=row, column=1, padx=10, pady=5, sticky="we")
    entry.config(width=10)

    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def select[T: str](master: tk.Toplevel, variable: tk.StringVar, row: int, values: list[T]) -> None:
    combobox = ttk.Combobox(master, textvariable=variable, values=typing.cast(list[str], values), state='readonly')
    combobox.grid(row=row, column=2, padx=0, pady=5, sticky="w")
    combobox.current(0)
    combobox.config(width=6)

def toggle(master: tk.Toplevel, label_text: str, variable: tk.BooleanVar, row: int, tip: str) -> None:
    checkbutton = tk.Checkbutton(master, text=label_text, variable=variable)
    checkbutton.grid(row=row, column=1, padx=10, pady=5, sticky="w")

    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def combobox(master: tk.Toplevel, label_text: str, variable: tk.Variable, row: int, values: list[typing.Any] | tuple[typing.Any, ...], tip: str | None = None, readonly: bool = False) -> None:
    label = tk.Label(master, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky='e')
    if readonly:
        combobox = ttk.Combobox(master, textvariable=variable, values=values, validate="key", validatecommand=(master.register(validate_numeric), "%P"), state='readonly')
    else:
        combobox = ttk.Combobox(master, textvariable=variable, values=values, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    combobox.grid(row=row, column=1, padx=10, pady=5, sticky='we')
    combobox.config(width=10)
    if tip:
        tooltip = tk.Label(master, text=tip, font=("Helvetica", 10), fg='gray', anchor="w", justify="left")
        tooltip.grid(row=row, column=2, padx=5, pady=5, sticky='w')

def dual_combobox(master: tk.Toplevel, label_text: str, variable: tk.Variable, divider: str, variable2: tk.Variable, row: int, values: list[typing.Any], values2: list[typing.Any], tip: str | None = None) -> None:
    label = tk.Label(master, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky='e')

    combobox = ttk.Combobox(master, textvariable=variable, values=values, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    combobox.grid(row=row, column=1, padx=10, pady=5, sticky='w')
    combobox.config(width=6)

    label2 = tk.Label(master, text=divider)
    label2.grid(row=row, column=1, padx=90, pady=5, sticky='we')

    combobox2 = ttk.Combobox(master, textvariable=variable2, values=values2, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    combobox2.grid(row=row, column=1, padx=10, pady=5, sticky='e')
    combobox2.config(width=6)
    if tip:
        tooltip = tk.Label(master, text=tip, font=("Helvetica", 10), fg='gray', anchor="w", justify="left")
        tooltip.grid(row=row, column=2, padx=5, pady=5, sticky='w')

audio_formats = "*.wav *.mp3 *.aac *.flac *.ogg *.opus *.wma *.m4a"
video_formats = "*.mp4 *.avi *.webm *.webp *.gif *.flv *.mkv *.mov *.wmv *.3gp"

def get_path_file(save=False, parent=None):
    formats = video_formats if save else audio_formats
    title = "Save video file" if save else "Select an audio file"
    filter_name = "Video files" if save else "Audio files"
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

    if "kde" in desktop and shutil.which("kdialog"):
        option = "--getsavefilename" if save else "--getopenfilename"
        command = ["kdialog", option, "", f"{filter_name} ({formats})"]
    elif shutil.which("zenity"):
        command = ["zenity", "--file-selection", f"--title={title}"]
        if save:
            command += ["--save", "--confirm-overwrite"]
        command += [
            f"--file-filter={filter_name} | {formats}",
            "--file-filter=All files | *",
        ]
    else:
        command = None

    if command:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip() or None
        if result.returncode == 1:
            return None

    is_root = parent is None
    dialog_parent = tk.Tk() if is_root else parent

    if is_root:
        dialog_parent.withdraw()

    try:
        if save:
            path = filedialog.asksaveasfilename(
                parent=dialog_parent,
                title=title,
                defaultextension=".mp4",
                filetypes=[("Video files", formats), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                parent=dialog_parent,
                title=title,
                filetypes=[("Audio files", formats), ("All files", "*.*")],
            )
        return path or None
    finally:
        if is_root:
            dialog_parent.destroy()


def file_input(parent: tk.Toplevel, label_text: str, row: int, path_var: tk.StringVar) -> tk.Entry:
    label = tk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

    entry = tk.Entry(parent, textvariable=path_var)
    entry.grid(row=row, column=1, columnspan = 2, padx=110, pady=5, sticky="w")
    entry.config(width=60)

    def browse():
        file_path = get_path_file(save = False)
        if file_path:
            if path_var is not None:
                path_var.set(file_path)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
            entry.xview_moveto(1)  #SHOW THE NAME OF THE FILE IN THE PATH

    button = tk.Button(parent, text="Browse", command=browse, width = 7)
    button.grid(row=row, column=1, padx=10, pady=5, sticky="w")

    return entry

def file_output(parent: tk.Toplevel, label_text: str, row: int, path_var: tk.StringVar) -> tk.Entry:
    label = tk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

    entry = tk.Entry(parent, textvariable=path_var)
    entry.grid(row=row, column=1, columnspan = 2, padx=110, pady=5, sticky="w")
    entry.config(width=60)

    def browse():
        file_path = get_path_file(save = True)
        if file_path:
            if path_var is not None:
                path_var.set(file_path)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
            entry.xview_moveto(1)  #SHOW THE NAME OF THE FILE IN THE PATH

    button = tk.Button(parent, text="Browse", command=browse, width = 7)
    button.grid(row=row, column=1, padx=10, pady=5, sticky="w")

    return entry

def back_button(root: tk.Tk, parent: tk.Toplevel) -> None:
    def go_back():
        root.deiconify() # VUELVE A MOSTRAR LA VENTANA PRINCIPAL
        for w in root.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.destroy()


    go_back_button = tk.Button(parent, text="↩", font=("TkFixedFont", 14), command=go_back)
    go_back_button.grid(row=0, column=0, padx=0, pady=0, sticky="nw")


def preview_toggle(master: tk.Toplevel, row: int, show_preview: typing.Callable[[tk.Toplevel], None]) -> None:
    preview_shown = tk.BooleanVar(value=True)

    def toggle_preview():
        if preview_shown.get():
            show_preview(master)
        else:
            if hasattr(master, "preview_window") and master.preview_window.winfo_exists():
                master.preview_window.destroy()

    preview_check = tk.Checkbutton(master, text="Live Preview\n (May be rough)", variable=preview_shown,command=toggle_preview)
    preview_check.grid(row=row, column=0, padx=5, pady=5, sticky="e")

    toggle_preview()
