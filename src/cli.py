import argparse
import threading
import time
import typing
from io import TextIOWrapper

import numpy as np
from bytesize import ByteSize
from tqdm import tqdm

import util
from backend import line_plot, scatter_plot, threshold
from backend.common import Visualizer

visualizers: dict[str, type[Visualizer]] = {
    "spectrum": line_plot.SpectrumVisualizer,
    "spectrum-db": line_plot.SpectrumdBVisualizer,
    "spectral-balance": line_plot.SpecBalanceVisualizer,
    "histogram": line_plot.HistogramVisualizer,
    "waveform": line_plot.WaveformVisualizer,
    "long-waveform": line_plot.LongWaveformVisualizer,
    "envelope": line_plot.EnvelopeVisualizer,
    "spectrogram": threshold.SpectrogramVisualizer,
    "oscilloscope": scatter_plot.OscilloscopeVisualizer,
    "polar-mono": scatter_plot.PolarMonoVisualizer,
    "polar-stereo": scatter_plot.PolarStereoVisualizer,
    "poincare": scatter_plot.PoincareVisualizer,
    "delay-embed": scatter_plot.DelayEmbedVisualizer,
    "recurrence": threshold.RecurrenceVisualizer,
    "false-chladni": threshold.FalseChladniVisualizer,
    "vectorscope": scatter_plot.VectorscopeVisualizer,
    "chladni": threshold.ChladniVisualizer,
}

main_cmd = argparse.ArgumentParser(prog="lsao")
subparsers = main_cmd.add_subparsers(required=True, dest="visualizer", title="modes")

for cmd_name, viz_ty in visualizers.items():
    sub_cmd = subparsers.add_parser(cmd_name, description=viz_ty.title, argument_default=argparse.SUPPRESS, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_cmd.add_argument("--measure-speed", action="store_true", default=False, help="Benchmark render/encode speed")
    sub_cmd.add_argument("--measure-alloc", action="store_true", default=False, help="Benchmark memory allocation")
    sub_cmd.add_argument("-i", "--input", dest="input_path", required=True, metavar="PATH", help="Path to input audio file")
    sub_cmd.add_argument("-o", "--output", dest="output_path", required=True, metavar="PATH", help="Path to output video file")

    default_obj = viz_ty()
    for param_name, (param_ty, metadata) in viz_ty.gather_params().items():
        arg_name = metadata.arg or f"--{param_name.replace("_", "-")}"
        if not isinstance(arg_name, list):
            arg_name = [arg_name]

        if isinstance(param_ty, typing.TypeAliasType):
            param_ty = param_ty.__value__

        if param_ty is bool:
            action = sub_cmd.add_argument(*arg_name, dest=param_name, help=metadata.description, action=argparse.BooleanOptionalAction)
        else:
            action = sub_cmd.add_argument(*arg_name, dest=param_name, help=metadata.description, metavar=param_ty.__name__.upper())

        if param_ty is int or param_ty is float:
            action.type = param_ty
        elif param_ty == tuple[int, int] or param_ty == tuple[float, float]:
            action.type = typing.get_args(param_ty)[0]
            action.nargs = 2
        elif typing.get_origin(param_ty) is typing.Literal:
            action.metavar = None
            # TODO: The capitalized multi-word values were fine up until now,
            # but definitely aren't suitable for CLI argument values.
            action.choices = typing.get_args(param_ty)

        if metadata.placeholder is not None:
            action.metavar = metadata.placeholder

        action.default = getattr(default_obj, param_name)

# A background thread to read ffmpeg progress updates and send them back to the UI thread progress bar
def capture_ffmpeg_progress(binary_stdout: typing.IO[bytes], iter: tqdm) -> None:
    keys_of_interest = {"speed", "bitrate", "total_size"}

    text_stdout = TextIOWrapper(binary_stdout)
    info = {}
    for line in text_stdout:
        key, val = line.strip().split("=", maxsplit=1)
        if key == "progress":
            iter.set_postfix(info)
            info = {}
            if val == "end":
                break
        elif key in keys_of_interest:
            if key == "total_size":
                key = "size"
                val = ByteSize(int(val))
            info[key] = val

def cli_main() -> None:
    args = main_cmd.parse_args()
    visualizer = visualizers[args.visualizer]()
    main_cmd.parse_args(namespace=visualizer)

    for param_name, (_ty, metadata) in visualizer.gather_params().items():
        # TODO: Report the error in a more friendly manner that indicates which argument is a problem
        metadata.validator(getattr(visualizer, param_name))

    if args.measure_alloc:
        import tracemalloc
        tracemalloc.start(5)

    t_start = time.perf_counter_ns()

    input_audio, sample_rate = util.read_audio_samples(args.input_path)
    audio = visualizer.prepare_audio(input_audio)

    t_loaded = time.perf_counter_ns()

    frame_sequence = visualizer.generate_video(audio, sample_rate)

    t_prepared = time.perf_counter_ns()

    ffmpeg_process = util.spawn_ffmpeg(
        resolution=visualizer.resolution,
        framerate=visualizer.framerate,
        crf=visualizer.compression,
        input_audio_path=args.input_path,
        output_video_path=args.output_path,
        capture_progress=True,
    )
    assert ffmpeg_process.stdin is not None

    t_first_frame = 0

    iter = tqdm(frame_sequence, delay=1, leave=False, unit="frames")
    threading.Thread(target=capture_ffmpeg_progress, args=(ffmpeg_process.stdout, iter), daemon=True).start()

    for frame_data in iter:
        t_first_frame = t_first_frame or time.perf_counter_ns()
        # Need to pack along the horizontal axis because ffmpeg expects each line to be a whole number of bytes
        ffmpeg_process.stdin.write(np.packbits(frame_data, axis=1))

    t_rendered = time.perf_counter_ns()

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    t_done = time.perf_counter_ns()

    if args.measure_alloc:
        _size, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        # Ignore memory used by modules (numpy is big and scipy is even bigger)
        snapshot = snapshot.filter_traces([tracemalloc.Filter(False, "<frozen importlib._bootstrap")])
        stats = snapshot.statistics("lineno")
        print("top 10 allocation sources:")
        for stat in stats[:10]:
            print(f"\t{stat}")

        print()
        print(f"peak memory usage: {peak / (1024 * 1024):.2f} MiB")

    if args.measure_speed:
        spans = {
            "total": t_done - t_start,
            "loading": t_loaded - t_start,
            "preparing": t_prepared - t_loaded,
            "first frame": t_first_frame - t_prepared,
            "rendering": t_rendered - t_first_frame,
            "encoding": t_done - t_rendered,
        }
        for name, duration_nanoseconds in spans.items():
            duration = duration_nanoseconds / 1_000_000_000
            print(f"{name}: {duration:.2f}")
