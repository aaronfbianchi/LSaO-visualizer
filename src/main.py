import os
import shutil
import sys


def main() -> None:
    use_bundled_ffmpeg(prefer_bundled=True)
    check_for_program("ffmpeg")
    check_for_program("ffprobe")

    if len(sys.argv) == 1 or "--gui" in sys.argv:
        import gui
        gui.gui_main()
    else:
        import cli
        cli.cli_main()

def check_for_program(program: str) -> None:
    path = shutil.which(program)
    if path is None:
        raise FileNotFoundError(f"{program} not found in $PATH or bundled with the app.")
    else:
        home_dir = os.path.expanduser("~")
        print(f"Using {program} @ {path.replace(home_dir, "~")}")

def use_bundled_ffmpeg(prefer_bundled: bool) -> None:
    # Search relative to the program, even if launched from another directory
    base_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    bundle_dir = os.path.join(base_dir, "ffmpeg", "bin")
    if prefer_bundled:
        os.environ["PATH"] = bundle_dir + os.pathsep + os.environ["PATH"]
    else:
        os.environ["PATH"] += os.pathsep + bundle_dir

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
