import os
import subprocess
import typing
from dataclasses import dataclass

import numpy as np

type VideoFrame = np.ndarray[tuple[int, int], np.dtype[np.bool]]

type MonoSamples[T: np.number] = np.ndarray[tuple[int], np.dtype[T]]
type StereoSamples[T: np.number] = np.ndarray[tuple[int, int], np.dtype[T]]
type AudioSamples[T: np.number] = MonoSamples[T] | StereoSamples[T]

@dataclass
class FrameSequence:
    length: int
    frames: typing.Generator[VideoFrame]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> typing.Iterator[VideoFrame]:
        return self.frames

def read_audio_samples(input_file: str) -> tuple[AudioSamples[np.int16], int]:
    cmd = ['ffmpeg', '-i', input_file, '-f', 's16le', '-']
    output = subprocess.check_output(cmd, stderr=subprocess.PIPE)

    cmd_probe = ['ffprobe', '-show_streams', '-select_streams', 'a:0', input_file]
    probe_output = subprocess.check_output(cmd_probe, stderr=subprocess.PIPE)
    probe_output = probe_output.decode('utf-8').split('\n')

    sampling_frequency = None
    num_channels = 1
    for line in probe_output:
        if line.startswith('sample_rate='):
            sampling_frequency = int(line.split('=')[1])
        elif line.startswith('channels='):
            num_channels = int(line.split('=')[1])

    if sampling_frequency is None:
        raise ValueError("Failed to extract sampling frequency from stream information")

    audio_samples = np.frombuffer(output, np.int16)

    if num_channels == 2:
        audio_samples = audio_samples.reshape(-1, num_channels)

    return audio_samples, sampling_frequency

type EncoderPreset = typing.Literal[
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
]

def spawn_ffmpeg(
    resolution: tuple[int, int],
    framerate: float,
    crf: float,
    input_audio_path: str,
    output_video_path: str,
    encoder_preset: EncoderPreset = "slow",
    silent: bool = False,
    capture_progress: bool = False,
):
    _, file_extension = os.path.splitext(output_video_path)

    video_input_args = [
        "-f", "rawvideo",
        "-s", f"{resolution[0]}x{resolution[1]}",
        "-pix_fmt", "monob",
        "-r", str(framerate),
        "-i", "-",
    ]

    video_output_args = []
    audio_input_args = []
    audio_output_args = []

    match file_extension:
        case ".gif":
            silent = True
        case ".3gp":
            video_output_args += ["-c:v", "mpeg4"]
        case ".webp":
            silent = True
            video_output_args += ["-loop", "0"]
        case ".avi":
            video_output_args += ["-c:v", "ffvhuff"]
        case ".wmv":
            video_output_args += ["-c:v", "wmv9screen"]
        case ".flv":
            video_output_args += ["-c:v", "libx264", "-crf", str(crf), "-preset", encoder_preset]
        case ".mov":
            video_output_args += ["-c:v", "libx264", "-crf", str(crf)]
        case ".mp4" | ".webm" | ".mkv" | ".264" | ".265":
            video_output_args += [
                "-preset", encoder_preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
            ]

    if not silent:
        audio_input_args += ["-i", input_audio_path]
        # TODO: What about -c:a copy? Could make sense for .mkv output.
        audio_output_args += ["-b:a", "320k"]
        match file_extension:
            case ".mov":
                audio_output_args = ["-c:a", "pcm_s24le"]
            case ".3gp":
                audio_output_args = ["-c:a", "aac"]

    logging_args = [
        # Hide the "banner", which gives information about the specific copy of FFmpeg being run.
        # For our purposes, this information is mostly noise.
        # Any case where someone might find it helpful for troubleshooting
        # is a case where that user is probably capable of getting that information themselves.
        "-hide_banner",
    ]
    stdout = None
    if capture_progress:
        stdout = subprocess.PIPE
        logging_args += [
            # Disable the default human-friendly progress updates, as they clash with LSaO's own progress bar.
            "-nostats",
            # Emit machine-readable progress updates to stdout.
            "-progress", "pipe:1",
            # (Other logs, e.g. stream info printed at start and end, go to stderr, and should be mostly fine to pass through as-is.)
        ]

    # Write progress messages to stdout
    # (to separate them from the stream info logged at the start and end, which goes to stderr)
    cmd = [
        "ffmpeg",
        *logging_args,
        *video_input_args,
        *audio_input_args,
        *video_output_args,
        *audio_output_args,
        # Overwrite existing output file
        # TODO: Ask first, like ffmpeg otherwise would? Might be annoying in GUI mode but CLI could just add a similar -y flag.
        "-y",
        output_video_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout)
