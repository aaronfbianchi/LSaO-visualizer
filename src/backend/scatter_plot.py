import math
import time
import typing
from typing import Annotated

import numpy as np

from util import (
    AudioSamples,
    FrameSequence,
    MonoSamples,
    StereoSamples,
    VideoFrame,
)

from . import validators
from .common import (
    MonoVisualizer,
    SquareVisualizer,
    StereoChannelName,
    StereoVisualizer,
    VizParam,
    apply_thickness,
    cubic_hermite,
    note_to_frequency,
    split_into_frames,
    rotate_osc_45,
)


class ScatterPlotVisualizer(SquareVisualizer):
    interpolate: Annotated[bool, VizParam("Interpolate", description="Use a spline to draw a curve connecting the dots.")] = True

    # This is separated out from plot_scatter_2d so that generate mode can do it for the entire file in one step.
    def prepare_scatter_2d[S: tuple[int, ...]](
        self,
        data: np.ndarray[S, np.dtype[np.float16]],
        *,
        res: tuple[int, int],
        scale: float,
    ) -> np.ndarray[S, np.dtype[np.float16]]:
        # Remap from [-1, 1] to [0, 1]
        data *= 0.5
        data *= scale #Apply scaling specified by the visualizer
        data += 0.5

        # Remap to video coordinates
        width, height = res
        data *= ((height - 1,), (width - 1,))
        return data

    def plot_scatter_2d(
        self,
        data: np.ndarray[tuple[int, int], np.dtype[np.float16]],
        *,
        res: tuple[int, int] | None = None,
        buffer: VideoFrame | None = None,
    ) -> VideoFrame:
        # Ensure the "coordinate" axis is last.
        # "Generate" mode sets this up in advance; "live" currently does not.
        if data.shape[0] == 2:
            data = data.T

        if self.interpolate:
            displacements = np.diff(data, axis=0)

            tangents = np.zeros_like(data)
            tangents[1:] += displacements
            tangents[:-1] += displacements
            tangents[1:-1] *= 0.5

            # Segment lengths of a linear spline connecting the points.
            # Dip into float32 because when you square a coordinate, it can max out float16
            distances = np.sqrt(np.vecdot(displacements, displacements, dtype=np.float32))

            # Locations of the points in a uniform 1D "parameter space".
            # The chosen parameter space moves along the linear spline at a constant speed.
            # For our purposes, our control points are dense enough that this can approximate
            # a parameter space that would have constant speed for a cubic spline.
            data_points = np.cumulative_sum(distances, include_initial=True)

            # Choose a uniformily-spaced set of sample points in parameter space.
            # Hopefully dense enough to create the illusion of a continuous curve, but no denser.
            num_points = math.ceil(data_points[-1] * 1.25)
            sample_points = np.linspace(0, data_points[-1], num_points)

            # Determine which segment of the spline each sample point belongs to.
            i1 = np.searchsorted(data_points, sample_points)
            i0 = np.fmax(i1 - 1, 0)

            # Determine the interpolation value for each sample point.
            # Once again, this uses the linear spline as an approximation.
            t0 = data_points[i0]
            t1 = data_points[i1]
            t = np.nan_to_num((sample_points - t0) / (t1 - t0), copy=False)

            data = cubic_hermite(t=t, p0=data[i0], p1=data[i1], m0=tangents[i0], m1=tangents[i1]).astype(np.float16)

        if buffer is None:
            width, height = res or self.resolution
            buffer = np.zeros((height, width), dtype=np.bool)

        coords = data.astype(np.int16)
        indices = np.ravel_multi_index(coords.T, buffer.shape, mode='clip')
        buffer.put(indices, 1, mode='clip')

        return apply_thickness(buffer, self.thickness)

    def generate_scatter_2d(
        self,
        arr: np.ndarray[tuple[int, int], np.dtype[np.float16]],
        sample_rate: int,
        *,
        scale: float,
    ) -> FrameSequence:
        data = self.prepare_scatter_2d(arr, res=self.resolution, scale=scale)
        frame_size = int(np.round(sample_rate / self.framerate))
        frames = split_into_frames(data, frame_size)

        # channel, frame, sample -> frame, sample, channel
        chunks = np.moveaxis(frames, 0, -1)

        num_frames = chunks.shape[0]

        def gen() -> typing.Generator[VideoFrame]:
            width, height = self.resolution
            frameData = np.empty((height, width), dtype=np.bool)
            for chunk in chunks:
                frameData.fill(0)
                yield self.plot_scatter_2d(chunk, buffer=frameData)

        return FrameSequence(num_frames, gen())

    def live_scatter_2d(
        self,
        arr: np.ndarray[tuple[int, int], np.dtype[np.float16]],
        resolution: tuple[int, int],
        *,
        scale: float,
    ) -> VideoFrame:
        data = self.prepare_scatter_2d(arr, res=resolution, scale=scale)
        return self.plot_scatter_2d(data, res=resolution)

class OscilloscopeVisualizer(StereoVisualizer, ScatterPlotVisualizer):
    title = "Oscilloscope Visualizer by Aaron F. Bianchi"

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio = audio/np.max(np.max(abs(abs(audio)))).astype(np.float16)
        audio = np.clip(audio, -1, 1)
        audio[1, :] *= -1
        return self.generate_scatter_2d(audio, sample_rate, scale=1.0)

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        self.interpolate = self.interpolate
        self.thickness = np.maximum(self.thickness, 1) # AVOID GOING TO THE BACKROOMS
        audio[1, :] *= -1
        return self.live_scatter_2d(audio, resolution, scale=1.0)

class VectorscopeVisualizer(OscilloscopeVisualizer):
    title = "Vectorscope Visualizer by Aaron F. Bianchi"
    def __init__(self) -> None:
        self.resolution = (720, 360)

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio = audio/np.max(np.max(abs(abs(audio)))).astype(np.float16)
        audio = np.clip(audio, -1, 1)*0.95

#         selector = (audio[1,:] + audio[0,:]) < 0
#         R_sel = np.where(selector, -audio[1,:], audio[1,:])
#         L_sel = np.where(selector, -audio[0,:], audio[0,:])
#
#         u = R_sel - L_sel
#         v = R_sel + L_sel
#
#         amp = np.maximum(np.abs(audio[1,:]), np.abs(audio[0,:]))
#         mag = np.hypot(u, v)
#
#         scale = np.where(mag > 0, amp / mag, 0.0)
#         audio[1,:] = u * scale
#         audio[0,:] = -v * scale
###################################################
#
        R = audio[1,:]
        L = audio[0,:]
#
#         eks = (R - L)*np.sign(R + L)*np.max((R,L))/np.sqrt(2*(R**2 + L**2))
#         why = np.abs(R + L)*np.max((R,L))/np.sqrt(2*(R**2 + L**2))
#
##########################################

#         x_sign = np.sign(R - L)
#         y_sign = np.sign(R + L)
#
#         eks = x_sign*np.sqrt(np.abs(np.abs(R)*R - np.abs(L)*L)/2)
#         why = y_sign*np.sqrt(np.abs(np.abs(R)*R + np.abs(L)*L)/2)



######################################
## ROMBIC VECTORSCOPE (ACTUALLY, THIS IS JUST THE NORMAL XY SCILLOSCOPE! :0)
        # eks = (L + R)/np.sqrt(2)
        # why = (L - R)/np.sqrt(2)


        ######################
        #  OTHER MATH

#         amp = np.maximum(R,L)
#         normLR = (R-L)/amp
#         arg = np.pi/4*normLR
#
#         eks = amp*np.sin(arg)
#         why = amp*np.cos(arg)

        ####################
        ## POLAR VECTORCOPE (GOOD ONE)

        cx = R*np.sqrt(1-L**2/2)
        cy = L*np.sqrt(1-R**2/2)

        eks = np.sign(cx+cy)*(cx-cy)*np.sqrt(2)/2
        why = np.abs(cx+cy)*np.sqrt(2)/2

        audio[1,:] = eks
        audio[0,:] = why
        ## *0.95 -> -1 -> *2 -> *(-1) IN THAT ORDER
        audio[0,:] *= -1.9
        audio[0,:] += 1

        audio = np.nan_to_num(audio)
        return self.generate_scatter_2d(audio, sample_rate, scale=1.0)

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        self.interpolate = self.interpolate
        self.thickness = np.maximum(self.thickness, 1) # AVOID GOING TO THE BACKROOMS
        R = audio[1,:]
        L = audio[0,:]

        cx = R*np.sqrt(1-L**2/2)
        cy = L*np.sqrt(1-R**2/2)

        eks = np.sign(cx+cy)*(cx-cy)*np.sqrt(2)/2
        why = np.abs(cx+cy)*np.sqrt(2)/2

        audio[1,:] = eks
        audio[0,:] = why
        ## *0.95 -> -1 -> *2 -> *(-1) IN THAT ORDER
        audio[0,:] *= -1.9
        audio[0,:] += 1

        audio = np.nan_to_num(audio)

        return self.live_scatter_2d(audio, resolution, scale=1.0)

class BasePolarVisualizer(ScatterPlotVisualizer):
    offset: Annotated[float, VizParam(
        "Offset",
        description="Set an offset. I don't know how to explain it. Just try and see.",
    )] = 0.5

    note: Annotated[str, VizParam(
        "Tuning",
        description="Set a note to tune the polar oscilloscope to.\nYou can enter the name of a note or its fundamental frequency in Hz.",
        validator=validators.tuning_frequency,
    )] = "C4"

    def __init__(self) -> None:
        super().__init__()
        self.live_polar_t = 0.0

class PolarMonoVisualizer(MonoVisualizer, BasePolarVisualizer):
    title = "Polar Visualizer by Aaron F. Bianchi"

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        # A4 ---> 2764.5
        polar_speed = note_to_polarSpeed(self.note)

        audio = (audio/np.max(np.max(abs(abs(audio))))) ##NORMALIZATION
        audio = np.clip(audio,-1,1)
        audio += self.offset ###################################################ADDING OFFSET
        audio = (audio/np.max(np.max(abs(abs(audio))))) ##NORMALIZATION AGAIN

        erre = audio.astype(np.float16)
        theta = (np.linspace(0, polar_speed*len(erre)/sample_rate, len(erre)) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision
        theta += np.pi/4 ## rotate_osc_45 GET REPLACED BY THIS
        audio = erre * (np.sin(theta), np.cos(theta))
        audio[1,:] *= -1 ##TO MAKE STEREO AND MONO POLAR IDENTICAL WHEN YOU FEED THEM A MONO FILE

        #audio = rotate_osc_45(audio)

        return self.generate_scatter_2d(audio, sample_rate, scale=0.95)

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        polar_speed = note_to_polarSpeed(self.note)
        audio += self.offset
        if self.offset != 0:
            audio /= abs(self.offset) + 1

        erre = audio.astype(np.float16)

        theta = (np.linspace(self.live_polar_t, self.live_polar_t + polar_speed*len(erre)/48000, len(erre)) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision
        theta += np.pi/4 ## rotate_osc_45 GET REPLACED BY THIS
        self.live_polar_t += polar_speed * len(erre)/48000 # UPDATE FRAME COUNTER
        #live_polar.t = live_polar.t%(2*np.pi)

        audio = erre * (np.sin(theta), np.cos(theta))
        audio[1,:] *= -1 ##TO MAKE STEREO AND MONO POLAR IDENTICAL WHEN YOU FEED THEM A MONO FILE

        #audio = rotate_osc_45(audio)

        return self.live_scatter_2d(audio, resolution, scale=0.95)

class PolarStereoVisualizer(StereoVisualizer, BasePolarVisualizer):
    title = "Stereo Polar Visualizer by Aaron F. Bianchi"

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        song = audio
        ##TRANSPOSITION AND NORMALIZATION
        song /= float(np.max(np.abs(song)))
        song = np.clip(song,-1,1)
        song += self.offset ###################################################ADDING OFFSET
        song /= float(np.max(np.abs(song))) ##NORMALIZATION AGAIN

        audio = typing.cast(StereoSamples, song)
        duration = audio.shape[1]

        # A4 ---> 2764.5
        polar_speed = note_to_polarSpeed(self.note)

        theta = (np.linspace(0, polar_speed*duration/sample_rate, duration) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision

        audio *= (np.sin(theta), np.cos(theta))

        audio = rotate_osc_45(audio)

        return self.generate_scatter_2d(audio, sample_rate, scale=0.95)

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        polar_speed = note_to_polarSpeed(self.note)
        if self.offset != 0:
            audio = audio + self.offset # for mysterious reasons, compound assignment does something different here
            audio /= abs(self.offset) + 1

        duration = audio.shape[1]
        theta = (np.linspace(self.live_polar_t, self.live_polar_t + polar_speed*duration/48000, duration) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision
        self.live_polar_t += polar_speed * duration/48000 # UPDATE FRAME COUNTER
        #live_polar.t = live_polar.t%(2*np.pi)

        audio *= (np.sin(theta), np.cos(theta))

        audio = rotate_osc_45(audio)

        return self.live_scatter_2d(audio, resolution, scale=0.95)

def note_to_polarSpeed(note):
    #A4 ---> 2764.55 (?????)
    frequency = note_to_frequency(note)*2

    return frequency

class PoincareVisualizer(MonoVisualizer, ScatterPlotVisualizer):
    title = "Poincaré Plot Visualizer by Aaron F. Bianchi"

    delay: Annotated[int, VizParam(
        "Delay",
        description="This can modify the shape of the scribble in cool ways.\nI don't know how to explain it. Just try and see.",
    )] = 10

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio /= np.max(np.abs(audio))
        audio = np.stack((np.roll(audio, -self.delay), -audio))
        return self.generate_scatter_2d(audio, sample_rate, scale=1.0)

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        self.thickness = np.maximum(self.thickness, 1) # AVOID GOING TO THE BACKROOMS
        self.interpolate = self.interpolate
        audio = np.stack((np.roll(audio, -self.delay), -audio))
        return self.live_scatter_2d(audio, resolution, scale=1.0)

# THREE channels!
# In stereo mode, the channels are [both, left, right].
# In other modes, the channels are all equal, aside from negation.
# The second and third channels are each shifted by their own "delay" value.
# Unfortunately, until numpy supports Literal[int] in array shapes, this type is identical to StereoSamples.
# Even so, it's clearer to give it a separate name.
type DelayEmbedSamples[T: np.number] = np.ndarray[tuple[int, int], np.dtype[T]]

class DelayEmbedVisualizer(ScatterPlotVisualizer):
    title = "Delay Embed Visualizer by Aaron F. Bianchi"

    channel: Annotated[StereoChannelName, VizParam("Channel", arg="--downmix", order=-2)] = "Both (Stereo)"

    delay: Annotated[tuple[int, int], VizParam(
        "Delays",
        description="This can modify the shape of the scribble in cool ways.",
        divider="&",
        # Tau: a time constant, and more specifically the
        placeholder=("τ₁", "τ₂"),
        values=[1, 5, 10, 15, 20, 30, 50, 100],
        values2=[1, 5, 10, 15, 20, 30, 50, 100],
        validator=validators.both_positive,
    )] = (10, 20)

    # Also known as alfa_p and beta_p
    angle: Annotated[tuple[float, float], VizParam(
        "Rotation",
        description="Initial angular positions (Yaw and Pitch) of the scribble in degrees.",
        divider="&",
        placeholder=("α₀", "β₀"),
        values=[-90,-45,-30,-15,0,15,30,45,90],
        values2=[-90,-45,-30,-15,0,15,30,45,90],
        ignored_in_live_mode=True,
    )] = (0, 15)

    # Also known as alfa_s and beta_s
    rotation_speed: Annotated[tuple[float, float], VizParam(
        "Rotation Speed",
        description="Rotation speed of the scribble in revolutions per second",
        divider="&",
        placeholder=("Δα", "Δβ"),
        values=[-1,-0.5,-0.25,-0.125,0,0.125,0.25,0.5,1],
        values2=[-1,-0.5,-0.25,-0.125,0,0.125,0.25,0.5,1],
    )] = (0.25, 0)

    def prepare_audio[T: (np.int16, np.float32)](self, input_audio: AudioSamples[T]) -> DelayEmbedSamples[np.float16]:
        mono: MonoSamples[T]
        stereo: StereoSamples[T] | None
        if input_audio.ndim == 1:
            mono = typing.cast(MonoSamples, input_audio)
            stereo = None
        else:
            assert input_audio.ndim == 2
            stereo = typing.cast(StereoSamples, input_audio)
            assert stereo.shape[1] == 2
            # TODO: Factor this transpose out of all three downmix functions and into read_audio_samples
            stereo = stereo.T
            match self.channel:
                case "Both (Stereo)": mono = np.mean(stereo, axis=0)
                case "Both (Merge to mono)": mono = np.mean(stereo, axis=0)
                case "Left": mono = stereo[0]
                case "Right": mono = stereo[1]

        # TODO: Figure out if this is the *intended* negation behavior
        if self.channel == "Both (Stereo)" and stereo is not None:
            x, y, z = -mono, stereo[0], stereo[1]
        else:
            x, y, z = -mono, -mono, -mono

        tau1, tau2 = self.delay
        return np.stack((x, np.roll(y, -tau1), np.roll(z, -tau2)), dtype=np.float16)

    def generate_video(self, audio: DelayEmbedSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio /= float(np.max(np.abs(audio)))
        audio = np.clip(audio, -1, 1)
        size_frame = int(np.round(sample_rate / self.framerate))
        data = split_into_frames(audio, size_frame)

        # xyz, frame, sample -> frame, xyz, sample
        chunks = np.moveaxis(data, 0, 1)
        num_frames = chunks.shape[0]

        def gen() -> typing.Generator[VideoFrame]:
            yaw, pitch = np.deg2rad(self.angle)
            yaw_rate, pitch_rate = self.rotation_speed
            omega = 2 * np.pi / self.framerate

            frameData = np.empty(self.resolution[::-1], dtype=np.bool)
            for chunk in chunks:
                yaw += omega * yaw_rate
                pitch += omega * pitch_rate

                rotated = rotate_points(chunk, yaw, pitch)
                rotated = rotated[1::-1] # Drop Z and, for some reason, swap X/Y
                coords = self.prepare_scatter_2d(rotated, res=self.resolution, scale=2/3)

                frameData.fill(0)
                yield self.plot_scatter_2d(coords, buffer=frameData)

        return FrameSequence(num_frames, gen())

    def live_preview(self, audio: DelayEmbedSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        alfa_s, beta_s = self.rotation_speed
        t = time.time()
        beta = 2*np.pi*beta_s*t
        alfa = 2*np.pi*alfa_s*t
        rotated = rotate_points(audio, alfa, beta)
        rotated = rotated[1::-1]
        return self.live_scatter_2d(rotated, resolution=resolution, scale=1.0)

def rotate_points(
    P: np.ndarray[tuple[int, int], np.dtype[np.float16]],
    yaw: float, pitch: float,
) -> np.ndarray[tuple[int, int], np.dtype[np.float16]]:
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R = Rx @ Ry
    return R @ P
