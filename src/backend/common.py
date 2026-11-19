import annotationlib
import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, overload

import numpy as np
from numpy.polynomial import Polynomial

from util import AudioSamples, FrameSequence, MonoSamples, StereoSamples, VideoFrame

from . import validators

type MonoChannelName = typing.Literal["Both (Merge to mono)", "Left", "Right"]
type StereoChannelName = typing.Literal["Both (Stereo)", "Both (Merge to mono)", "Left", "Right"]

@dataclass
class VizParam[T]:
    name: str
    description: str | None = None
    values: list[T] | None = None
    values2: list[T] | None = None
    ignored_in_live_mode: bool = False
    divider: str = ""
    # Position of this parameter in the UI. Negative numbers go at the start, positive at the end.
    # The plan:
    # - Input/output path
    # - Channel, framerate, resolution
    # - (Everything else)
    # - Drawing style, thickness, compression
    order: float = 0
    # TODO: The tuple hack has become a problem.
    validator: Callable[[typing.Any], None] = lambda _: None
    arg: str | list[str] | None = None
    placeholder: str | tuple[str, ...] | None = None

class Visualizer[C: AudioSamples[np.float16]](ABC):
    title: ClassVar[str]

    resolution: Annotated[tuple[int, int], VizParam(
        "Resolution",
        description="Width × Height. Even numbers.",
        divider="×",
        placeholder=("WIDTH", "HEIGHT"),
        order=-1,
        values=[480, 720, 960, 1024, 1280, 1366, 1440, 1080, 1920, 2560, 3840],
        values2=[240, 360, 480, 540, 640, 720, 768, 960, 1080, 1440, 1600, 1920, 2160],
        ignored_in_live_mode=True,
        validator=validators.resolution,
    )] = (1920, 540)

    thickness: Annotated[int, VizParam(
        "Thickness",
        description="Will duplicate the curve one pixel to the right and up. Whole number.\nWill make the render slower the higher you go.",
        placeholder="PIXELS",
        order=2,
        values=[1, 2, 3, 4, 5],
        validator=validators.positive_integer,
    )] = 1

    framerate: Annotated[float, VizParam(
        "Frame rate",
        description="Frames per second",
        placeholder="FPS",
        order=-2,
        values=[23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120],
        ignored_in_live_mode=True,
        validator=validators.positive,
    )] = 60

    compression: Annotated[float, VizParam(
        "Video Compression",
        description="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.",
        placeholder="CRF",
        order=3,
        ignored_in_live_mode=True,
        validator=validators.non_negative,
    )] = 0

    @classmethod
    def gather_params(cls) -> dict[str, tuple[type, VizParam]]:
        params: list[tuple[str, tuple[type, VizParam]]] = []

        # Search a visualizer class, and all of its base classes...
        for ty in inspect.getmro(cls):
            # ...for fields with static type annotations...
            for name, field_type in annotationlib.get_annotations(ty).items():
                # ...that are wrapped in typing.Annotated[]...
                if typing.get_origin(field_type) is typing.Annotated:
                    actual_type, *metadata = typing.get_args(field_type)
                    # ...and include a VizParam object in the metadata.
                    for annotation in metadata:
                        if isinstance(annotation, VizParam):
                            params.append((name, (actual_type, annotation)))

        params.sort(key=lambda kv: kv[1][1].order)
        return dict(params)

    @abstractmethod
    def prepare_audio[T: (np.int16, np.float32)](self, input_audio: AudioSamples[T]) -> C: ...

    @abstractmethod
    def generate_video(self, audio: C, sample_rate: int) -> FrameSequence: ...

    @abstractmethod
    def live_preview(self, audio: C, resolution: tuple[int, int]) -> VideoFrame: ...

# A visualizer that expects single-channel audio. Stereo input will be downmixed.
class MonoVisualizer(Visualizer[MonoSamples[np.float16]]):
    channel: Annotated[MonoChannelName, VizParam("Channel", arg="--downmix", order=-2)] = "Both (Merge to mono)"

    def prepare_audio[T: (np.int16, np.float32)](self, input_audio: AudioSamples[T]) -> MonoSamples[np.float16]:
        if input_audio.ndim == 1:
            return typing.cast(MonoSamples, input_audio).astype(np.float16)

        assert input_audio.ndim == 2
        input_audio = typing.cast(StereoSamples, input_audio)
        # Confirm that the channel axis is currently at the end (Should be changed eventually)
        assert input_audio.shape[1] == 2
        input_audio = input_audio.T

        audio: MonoSamples[np.int16]
        match self.channel:
            case "Both (Merge to mono)": audio = np.mean(input_audio, axis=0)
            case "Left": audio = input_audio[0]
            case "Right": audio = input_audio[1]
        return audio.astype(np.float16)

# A visualizer that expects stereo audio, and requires it for meaningful output.
class StereoVisualizer(Visualizer[StereoSamples[np.float16]]):
    def prepare_audio[T: (np.int16, np.float32)](self, input_audio: AudioSamples[T]) -> StereoSamples[np.float16]:
        assert input_audio.ndim == 2, f"Input audio is not stereo.\nExpected a 2D array, but got a {input_audio.ndim}D array."
        input_audio = typing.cast(StereoSamples, input_audio)
        # Confirm that the channel axis is currently at the end (Should be changed eventually)
        assert input_audio.shape[1] == 2, f"Input audio is not stereo.\nExpected 2 channels, but got {input_audio.shape[1]} channels."
        return input_audio.T.astype(np.float16)

# A visualizer that accepts two-channel audio, but can produce meaningful output even if both channels match.
class OptionalStereoVisualizer(StereoVisualizer):
    channel: Annotated[StereoChannelName, VizParam("Channel", arg="--upmix", order=-2)] = "Both (Stereo)"

    def prepare_audio[T: (np.int16, np.float32)](self, input_audio: AudioSamples[T]) -> StereoSamples[np.float16]:
        mono: MonoSamples[np.int16]

        if input_audio.ndim == 1:
            mono = typing.cast(MonoSamples, input_audio)
        else:
            assert input_audio.ndim == 2
            stereo = typing.cast(StereoSamples, input_audio)
            # Confirm that the channel axis is currently at the end (Should be changed eventually)
            assert stereo.shape[1] == 2
            stereo = stereo.T

            match self.channel:
                case "Both (Stereo)": return stereo.astype(np.float16)
                case "Both (Merge to mono)": mono = np.mean(stereo, axis=0)
                case "Left": mono = stereo[0]
                case "Right": mono = stereo[1]

        return np.stack((mono, mono)).astype(np.float16)

class SquareVisualizer(Visualizer):
    # A small hack to replace the defaults and suggested values from the base Visualizer with square versions.
    resolution: Annotated[tuple[int, int], VizParam(
        "Resolution",
        description="Width × Height. Even numbers.",
        divider="×",
        placeholder=("WIDTH", "HEIGHT"),
        # The crux of this hack: a slightly more positive order.
        # This makes the definition show up later in the list, which makes it replace the original definition in the dict returned by gather_params.
        order=-0.9,
	    values = [240, 360, 480, 540, 640, 720, 768, 960, 1080, 1440, 1600, 1920, 2160],
	    values2 = [240, 360, 480, 540, 640, 720, 768, 960, 1080, 1440, 1600, 1920, 2160],
        ignored_in_live_mode=True,
        validator=validators.resolution,
    )]

    def __init__(self) -> None:
        self.resolution = (720, 720)

def rotate_osc_45(audio):
    sqrt2_over_2 = (np.sqrt(2) / 2).astype(np.float16)
    audioL = (audio[1] + audio[0]) * sqrt2_over_2
    audioR = (-audio[1] + audio[0]) * sqrt2_over_2
    audio[0] = audioL
    audio[1] = audioR
    return audio

def apply_thickness(frameData: VideoFrame, thickness: int) -> VideoFrame:
    if thickness > 1:
        for _ in range(thickness - 1):
            shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
            shifted[-1, :] = np.zeros(shifted.shape[1]) ## CLEARS BOTTOM ROW
            shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
            shifted2[:, -1] = np.zeros(shifted.shape[0]) ## CLEARS LAST COLUMN
            frameData = frameData | shifted | shifted2
    return frameData

def pick_channel[T: np.number](song: AudioSamples[T], channel: MonoChannelName) -> MonoSamples[T]:
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            return np.transpose(np.mean(song, axis = 0))
        elif channel == "Left":
            return np.transpose(song[0,:])
        elif channel == "Right":
            return np.transpose(song[1,:])
    else:
        assert song.ndim == 1
        return np.transpose(song) # type: ignore

# Split an array (along its innermost axis) into slices of length `frame_size`.
# If the input length is not a multiple of the frame size, the end will be padded with zeros to complete the final frame.
# Supposing a sample count N and frame size F:
# (N,) -> (ceil(N/F), F)
# (2, N) -> (2, ceil(N/F), F)
@overload
def split_into_frames[T: np.dtype](arr: np.ndarray[tuple[int], T], frame_size: int) -> np.ndarray[tuple[int, int], T]: ...

@overload
def split_into_frames[T: np.dtype](arr: np.ndarray[tuple[int, int], T], frame_size: int) -> np.ndarray[tuple[int, int, int], T]: ...

def split_into_frames[S: tuple[int, ...], T: np.dtype](arr: np.ndarray[S, T], frame_size: int) -> np.ndarray[tuple[int, ...], T]:
    original_len = arr.shape[-1]
    num_frames = int(np.ceil(original_len / frame_size))

    padded_len = num_frames * frame_size
    if padded_len != original_len:
        # "Complete" the last frame so that there's the right amount of data for the reshape
        pad_amount = padded_len - original_len
        pad_widths = [(0, 0) for _ in range(arr.ndim)]
        pad_widths[-1] = (0, pad_amount)
        arr = np.pad(arr, pad_widths, mode='edge')

    new_shape = arr.shape[:-1] + (num_frames, frame_size)
    return arr.reshape(new_shape)

def note_to_frequency(note: str) -> float:
    note = note.strip().capitalize()  # c#4 -> C#4

    if note.lstrip('-').replace('.', '', 1).isdigit():
        return float(note)

    note_to_midi = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                    'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6,
                    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
                    'B': 11, 'Cb': 11, 'B#': 0}

    if len(note) < 2:  # MUY CORTITO
        note_name, octave_str = "C", "4"
    else:
        note_name, octave_str = note[:-1], note[-1]

    # Validate note name
    if note_name not in note_to_midi:
        note_name, octave_str = "C", "4"

    # Validate octave
    try:
        octave = int(octave_str)
    except ValueError:
        octave = 4

    midi_note = note_to_midi[note_name] + (octave + 1) * 12
    frequency = 440 * (2 ** ((midi_note - 69) / 12))
    #print(frequency)
    return frequency

hermite_terms = [
    Polynomial([1, 0, -3, 2]), # p0
    Polynomial([0, 1, -2, 1]), # m0
    Polynomial([0, 0, 3, -2]), # p1
    Polynomial([0, 0, -1, 1]), # m1
]
def cubic_hermite[S: tuple[int, ...]](
    *,
    p0: np.ndarray[S, np.dtype[np.number]],
    m0: np.ndarray[S, np.dtype[np.number]],
    p1: np.ndarray[S, np.dtype[np.number]],
    m1: np.ndarray[S, np.dtype[np.number]],
    t: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[S, np.dtype[np.float32]]:
    interp_terms = np.asarray([f(t) for f in hermite_terms])
    control_terms = np.asarray([p0, m0, p1, m1])

    while interp_terms.ndim < control_terms.ndim:
        interp_terms = np.expand_dims(interp_terms, -1)

    out: np.ndarray[S, np.dtype[np.float32]] = np.empty_like(p0, dtype=np.float32) # ty: ignore
    return np.sum(interp_terms * control_terms, axis=0, out=out)
