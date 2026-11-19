from ast import MatchClass
import time
import typing
from typing import Annotated

import numpy as np

from util import FrameSequence, MonoSamples, StereoSamples, VideoFrame

from . import validators
from .common import (
    MonoVisualizer,
    OptionalStereoVisualizer,
    SquareVisualizer,
    VizParam,
    apply_thickness,
    note_to_frequency,
    split_into_frames,
)
from .line_plot import trim_fft


class ThresholdVisualizer(OptionalStereoVisualizer, SquareVisualizer):

    def apply_threshold(
        self,
        values: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        out: VideoFrame | None = None,
    ) -> VideoFrame:
        if out is None:
            out = np.empty_like(values, dtype=np.bool)

        if self.threshold < 0:
            np.greater(values, -self.threshold, out=out)
        else:
            np.less(values, self.threshold, out=out)

        return out

class RecurrenceVisualizer(ThresholdVisualizer):
    title = "Recurrence Plot Visualizer by Aaron F. Bianchi"
    threshold: Annotated[float, VizParam(
        "Threshold",
        description="Higher values will increase the amount of black.",
    )] = 0.05
    channel = "Both (Merge to mono)"

    note: Annotated[str, VizParam(
        "Tuning",
        description="Set a note to tune the recurrence plot to.\nYou can enter the name of a note or its fundamental frequency in Hz.",
        validator=validators.tuning_frequency,
        ignored_in_live_mode=True,
    )] = "C2"

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio /= np.max(audio) # normalize
        freq_tune = note_to_frequency(self.note)

        # See waveform mode for commentary on these calculations.
        duration = audio.shape[-1] / sample_rate
        speed = sample_rate / freq_tune
        desired_frame_count = duration * freq_tune
        actual_frame_count = int(duration * self.framerate)
        frame_indices = np.round(np.linspace(0, desired_frame_count - 1, actual_frame_count))

        full_left, full_right = np.unstack(audio)
        full_range = np.arange(len(full_left))

        def gen() -> typing.Generator[VideoFrame]:
            frameData = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.bool)
            for i in frame_indices:
                start = i * speed
                end = start + speed - 1
                end = np.fmin(end, full_range[-1])
                left = np.interp(np.linspace(start, end, self.resolution[0]), full_range, full_left)
                right = np.interp(np.linspace(start, end, self.resolution[1]), full_range, full_right)

                values = np.abs(left[np.newaxis, :] - right[:, np.newaxis])
                self.threshold *= -1
                self.apply_threshold(values, out=frameData)
                yield apply_thickness(frameData, self.thickness)

        return FrameSequence(len(frame_indices), gen())

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        left, right = np.unstack(audio)
        n = len(left)
        left = np.interp(np.linspace(0, n - 1, resolution[0]), np.arange(n), left)
        right = np.interp(np.linspace(0, n - 1, resolution[1]), np.arange(n), right)

        values = np.abs(left[np.newaxis, :] - right[:, np.newaxis])
        self.threshold *= -1
        frameData = self.apply_threshold(values)
        return apply_thickness(frameData, self.thickness)

type FalseChladniMode = typing.Literal["Sine", "Cosine", "Tangent", "Cotangent", "Secant", "Cosecant"]

class FalseChladniVisualizer(ThresholdVisualizer):
    title = "False Chladni Plate Visualizer by Aaron F. Bianchi"
    threshold: Annotated[float, VizParam(
        "Threshold",
        description="Higher values will increase the amount of white.",
    )] = 0.5

    mode: Annotated[FalseChladniMode, VizParam("Mode", description="Try these for different shapes in your video")] = "Cosine"

    zoom: Annotated[float, VizParam(
        "Zoom",
        description="Zoom in the Chladni plate.",
        validator=validators.positive,
    )] = 1000

    smoothing: Annotated[float, VizParam(
        "Smoothing",
        description="The higher this value, the smoother the visualization. From 0 to 1.",
        validator=validators.between_0_1_inclusive,
    )] = 0.2

    def __init__(self) -> None:
        super().__init__()
        self.env_past = np.zeros(2)

    def do_trig(
        self,
        env: np.ndarray[tuple[int], np.dtype[np.floating]],
        a: float, b: float,
        xaxis: np.ndarray[tuple[int], np.dtype[np.floating]],
        yaxis: np.ndarray[tuple[int], np.dtype[np.floating]],
        out: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        func: typing.Callable[[np.ndarray], np.ndarray[tuple[int], np.dtype[np.float64]]]
        recip: bool
        match self.mode:
            case "Sine": func, recip = np.sin, False
            case "Cosine": func, recip = np.cos, False
            case "Tangent": func, recip = np.tan, False
            case "Cotangent": func, recip = np.tan, True
            case "Secant": func, recip = np.cos, True
            case "Cosecant": func, recip = np.sin, True

        trgXL = func(xaxis / env[0] * a)[np.newaxis, :]
        trgXR = func(xaxis / env[1] * b)[np.newaxis, :]
        trgYL = func(yaxis / env[0] * a)[:, np.newaxis]
        trgYR = func(yaxis / env[1] * b)[:, np.newaxis]
        if recip:
            np.reciprocal(trgXL, out=trgXL)
            np.reciprocal(trgXR, out=trgXR)
            np.reciprocal(trgYL, out=trgYL)
            np.reciprocal(trgYR, out=trgYR)

        return np.abs(a * trgXL * trgYR + b * trgXR * trgYL, out=out)

    def finalize(self, values: np.ndarray[tuple[int, int], np.dtype[np.floating]], out: VideoFrame) -> VideoFrame:
        mid_y, mid_x = values.shape

        # It's important to use a negative index when slicing the second half here,
        # as the halved dimensions will be rounded down if the full dimensions are odd.
        # Validation should prevent odd resolutions for video generation, but perhaps not for the live preview.

        bottom = out[-mid_y:]
        self.apply_threshold(values, bottom[:, -mid_x:])

        bottom[:, :mid_x] = np.fliplr(bottom[:, -mid_x:]) # Mirror bottom left <- bottom right
        out[:mid_y] = np.flipud(bottom)                   # Mirror top <- bottom
        return out

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        #if fil: #HIGH-PASS FILTER FOR HAVING MORE TRANSIENTS
        N = 5
        h = np.cos(np.linspace(0,2*np.pi,N)) - 1
        h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
        audio = np.apply_along_axis(lambda row: np.convolve(row, h, mode = 'same'), 1, audio)

        ##TRANSPOSITION AND NORMALIZATION
        audio /= float(np.max(np.abs(audio)))

        duration = audio.shape[1]/sample_rate
        size_frame = int(np.round(sample_rate/self.framerate))
        audioShaped = split_into_frames(audio, size_frame)
        num_frames = audioShaped.shape[1]

        res_width, res_height = self.resolution
        xaxis = np.linspace(1/res_width , res_width/self.zoom , int(res_width/2)).astype(np.float32) #solo el primer cuadrante
        yaxis = np.linspace(1/res_height , res_height/self.zoom , int(res_height/2)).astype(np.float32) #solo el primer cuadrante
        xaxis *= np.pi
        yaxis *= np.pi

        exp_filter = np.sqrt(self.smoothing)

        def gen() -> typing.Generator[VideoFrame]:
            frame = np.empty((res_height, res_width), dtype=np.bool)
            values = np.empty((res_height // 2, res_width // 2), dtype=np.float32)
            env_past = np.zeros(2)
            for i in range(num_frames):
                env_now = 10 * np.max(audioShaped[:, i, :], axis=1)
                env = env_now * (1 - exp_filter) + env_past * exp_filter
                env_past = env

                a = 5*(2+np.sin(1+i/num_frames*duration))
                b = 5*(2-np.cos(1+i/num_frames*duration))
                self.do_trig(env, a, b, xaxis, yaxis, out=values)
                self.finalize(values, out=frame)
                yield apply_thickness(frame, self.thickness)

        return FrameSequence(num_frames, gen())

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        #if fil: #HIGH-PASS FILTER FOR HAVING MORE TRANSIENTS
        N = 5
        h = np.cos(np.linspace(0,2*np.pi,N)) - 1
        h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
        audio[0] = np.convolve(audio[0], h, mode = 'same')
        audio[1] = np.convolve(audio[1], h, mode = 'same')

        res_width, res_height = resolution
        xaxis = np.linspace(1/res_width , res_width/self.zoom , int(res_width/2)).astype(np.float32) #solo el primer cuadrante
        yaxis = np.linspace(1/res_height , res_height/self.zoom , int(res_height/2)).astype(np.float32) #solo el primer cuadrante
        xaxis *= np.pi
        yaxis *= np.pi

        exp_filter = np.sqrt(self.smoothing)

        env_now = 10 * np.max(audio, axis=1) #DETERMINAR LA ENVOLVENTE
        env = env_now * (1 - exp_filter) + self.env_past * exp_filter
        self.env_past = env

        a = 5*(2+np.sin(1+time.time()))
        b = 5*(2-np.cos(1+time.time()))

        values = self.do_trig(env, a, b, xaxis, yaxis, out=np.empty((res_height // 2, res_width // 2)))
        frame = self.finalize(values, out=np.empty((res_height, res_width), dtype=np.bool))
        return apply_thickness(frame, self.thickness)

class SpectrogramVisualizer(MonoVisualizer):
    title = "Linear Spectrogram Visualizer by Aaron F. Bianchi"
    threshold: Annotated[float, VizParam(
        "Threshold",
        description="Higher values will increase the amount of black.",
    )] = 0.5

    ## HIGH PASS FILTER BECAUSE OTHERWISE YOU WOULDN'T SEE A THING LMAO
    highpass: Annotated[int, VizParam(
        "Mids & Highs Boost",
        description="This will make the mids and highs more visible. Positive whole number.",
        validator=validators.positive_integer,
    )] = 10

    time: Annotated[float, VizParam(
        "Time",
        description="Speed of the thing moving to the left. Positive number.",
        validator=validators.positive,
        ignored_in_live_mode=True,
    )] = 2

    pad_end: Annotated[bool, VizParam(
        "Pad End",
        description="Extend the audio with silence so that the video ends with a flat line.",
        ignored_in_live_mode=True,
    )] = False

    frequency_limits: Annotated[tuple[int, int], VizParam(
        "Frequency Limits",
        description="Lower and higher frequency limits in Hz, respectively.\nFrom 1Hz to half the sample rate of the audio.",
        divider="-",
        placeholder=("MIN", "MAX"),
        values=[1,20,300],
        values2=[600,1000,2000,5000,13000,20000],
        validator=validators.positive_integer_range,
    )] = (20, 13000)

    detail: Annotated[int, VizParam(
        "Vertical Detail",
        description="Level of vertical detail of the spectrogram.",
        ignored_in_live_mode=True,
    )] = 400

    def __init__(self) -> None:
        super().__init__()
        self.oldSpecLive = np.zeros((48000//60, 60*5)).astype('bool') #800 samples per frame, 5 seconds

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        res_width, res_height = self.resolution
        num_samples = audio.shape[-1]
        duration_seconds = num_samples / sample_rate
        num_frames = int(np.ceil(duration_seconds * self.framerate))
        if self.pad_end:
            num_frames += int(self.time * self.framerate) + 1

        high_pass_filter = np.cos(np.linspace(0, 2*np.pi, 2*self.highpass+1)) - 1
        high_pass_filter[self.highpass] -= sum(high_pass_filter)
        audio = np.convolve(audio, high_pass_filter, mode = 'same')

        audioShaped = split_into_frames(audio, self.detail)
        audioShaped *= np.hamming(self.detail)

        fsong = abs(np.fft.rfft(audioShaped, axis = 1))
        fsong /= np.max(fsong)

        fsong_trim = trim_fft(fsong, self.frequency_limits, sample_rate)

        fsong_bool = fsong_trim.T > self.threshold**6 ##THRESHOLD REMAPPED

        # Scale to output resolution (nearest neighbor)
        full_width = int(res_width * duration_seconds / self.time)
        data_height, data_width = fsong_bool.shape
        x_axis = np.linspace(0, data_width - 1, full_width).round().astype(np.uint32)
        y_axis = np.linspace(0, data_height - 1, res_height).round().astype(np.uint32)
        fsong_bool = fsong_bool[np.ix_(y_axis, x_axis)]

        fsong_bool = np.pad(fsong_bool, ((0, 0), (res_width, res_width if self.pad_end else 0)))
        fsong_bool = np.flipud(fsong_bool)

        time_indices = (np.arange(num_frames) * res_width/self.time/self.framerate).round().astype(np.int32)
        return FrameSequence(num_frames, (fsong_bool[:, i:i+res_width] for i in time_indices))


    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        res_width, res_height = resolution
        frameData = np.zeros((res_height, res_width), dtype=np.bool)

        high_pass_filter = np.cos(np.linspace(0, 2*np.pi, 2*self.highpass+1)) - 1
        high_pass_filter[self.highpass] -= sum(high_pass_filter)
        audio = np.convolve(audio, high_pass_filter, mode = 'same')

        audio *= np.hamming(audio.shape[0])

        fsong = abs(np.fft.rfft(audio))

        fsong_trim = trim_fft(fsong, self.frequency_limits, 48000)
        fsong_bool = fsong_trim.T > 500*self.threshold**6 ##THRESHOLD REMAPPED

        indices = np.linspace(0, len(fsong_trim) - 1, 48000//60).round().astype('int')
        fsong_scalled = fsong_bool[indices]

        self.oldSpecLive = np.roll(self.oldSpecLive, -1, axis=1)
        self.oldSpecLive[:,-1] = np.flipud(fsong_scalled)

        # Scale to output resolution (nearest neighbor)
        data_height, data_width = self.oldSpecLive.shape
        x_axis = np.linspace(0, data_width - 1, res_width).round().astype(np.uint32)
        y_axis = np.linspace(0, data_height - 1, res_height).round().astype(np.uint32)
        frameData = self.oldSpecLive[np.ix_(y_axis, x_axis)]

        frameData = apply_thickness(frameData, self.thickness)

        return frameData

class ChladniVisualizer(MonoVisualizer):
    title = "Chladni Plate Visualizer by Aaron F. Bianchi"
    threshold: Annotated[float, VizParam(
        "Threshold",
        description="Higher values will increase the amount of black.",
    )] = 10

    origin: Annotated[tuple[float, float], VizParam(
        "Sound Origin",
        description="These are the coordinates of the origin of the sound in the plate.\nFrom 0 to 1. (0; 0) is the center of the plate.",
        divider=";",
        placeholder=("MIN", "MAX"),
        values=[0,0.25,0.33,0.5,0.66,0.75,1],
        values2=[0,0.25,0.33,0.5,0.66,0.75,1],
        validator=validators.both_between_0_1_inclusive,
    )] = (0.25, 0.25)

    fixed_point: Annotated[tuple[float, float], VizParam(
        "Fixed Point",
        description="Set the coordinates for a fixed point. From 0 to 1.\nIf you don't want it, set it to the same coordinates as the sound origin.",
        divider=";",
        placeholder=("MIN", "MAX"),
        values=[0,0.25,0.33,0.5,0.66,0.75,1],
        values2=[0,0.25,0.33,0.5,0.66,0.75,1],
        validator=validators.both_between_0_1_inclusive,
    )] = (0.25, 0.25)

    k: Annotated[float, VizParam(
        "Rigidity",
        description="Rigidity of the plate.",
    )] = 20

    vol: Annotated[float, VizParam(
        "Volume Multiplier",
        description="Volume of the audio to be fed into the plate.",
    )] = 1

    upscale: Annotated[int, VizParam(
        "Grain Size",
        description="The bigger, the easier for your computer to run this. Positive whole number.",
    )] = 8

    time_res: Annotated[int, VizParam(
        "Time Size",
        description="The bigger, the easier for your computer to run this. Positive whole number.",
    )] = 2

    def __init__(self) -> None:
        self.resolution = (720, 720)

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio = audio/max(audio)
        res_width, res_height = self.resolution
        size_frame = int(sample_rate//self.framerate)
        num_frames = int(len(audio)/size_frame) + 1
        audio = np.pad(audio, (0, size_frame*num_frames - len(audio)))
        # print(len(audio)%size_frame, size_frame)
        # print(len(audio))
        # print(size_frame,num_frames)

        origin_x, origin_y = self.origin
        fixed_x, fixed_y = self.fixed_point

        audio_shaped = np.reshape(audio, (num_frames, size_frame))
        # frames = np.zeros((num_frames, res_width, res_height)).astype('bool')

#         for f in range(num_frames):
#             frames[f, :, :] = do_chladni_frame(sample_rate, audio_shaped[f, :], res_width, res_height, self.k, self.m, self.vol, self.upscale, self.time_res, self.thickness, self.threshold).astype('bool')

        return FrameSequence(num_frames, (do_chladni_frame(sample_rate, audio_shaped[f, :], res_width, res_height, self.k, self.vol, self.upscale, self.time_res, self.thickness, self.threshold, origin_x, origin_y, fixed_x, fixed_y) for f in range(num_frames)))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        sample_rate = 48000
        res_width, res_height = resolution
        origin_x, origin_y = self.origin
        fixed_x, fixed_y = self.fixed_point

        return do_chladni_frame(sample_rate, audio, res_width, res_height, self.k, self.vol, self.upscale, self.time_res, self.thickness, self.threshold, origin_x, origin_y, fixed_x, fixed_y)

def do_chladni_frame(sample_rate, audio, res_width, res_height, k, vol, upscale, time_res, thickness, threshold, origin_x, origin_y, fixed_x, fixed_y):
    # t = time.time()
    audio = np.concat([audio, audio]) ## WE ARE ONLY GRABBING THE SECOND CYCLE OF THE VIBRATION
    width_plate = int(res_width//2//upscale)
    height_plate = int(res_height//2//upscale)
    window = int(len(audio)//time_res)
    dt = 1/sample_rate

    origin_x = int(np.max((np.min(((width_plate*origin_x), width_plate-1)), 0)))
    origin_y = int(np.max((np.min(((height_plate*origin_y), height_plate-1)), 0)))
    fixed_x = int(np.max((np.min(((width_plate*fixed_x), width_plate-1)), 0)))
    fixed_y = int(np.max((np.min(((height_plate*fixed_y), height_plate-1)), 0)))

    k *= 1
    m = 0.000001
    vol *= 0.001
    pos_multiplier = 10000

    # THESE FORCE MULTIPLIERS EXIST BECAUSE WE ARE ONLY RENDERING ONE QUADRANT OF THE PICTURE
    # HENCE, THE ROW=0 AND COLUMN=0 NEED TO TAKE DOUBLE THE FORCE OF THEIR NEIGHBOURS (THE THING IS SYMMETRICAL)
    f_mul_1 = np.ones((width_plate - 1,height_plate))
    f_mul_2 = np.ones((width_plate,height_plate - 1))
    # f_mul_3 = np.ones((width_plate - 1,height_plate - 1))
    f_mul_1[0, :] = 2
    f_mul_2[:, 0] = 2
    # f_mul_3[0, :] = 2
    # f_mul_3[:, 0] = 2

    posit = np.zeros((width_plate,height_plate))
    veloc = np.zeros((width_plate,height_plate))
    accel = np.zeros((width_plate,height_plate))
    force = np.zeros((width_plate,height_plate))
    posit_acum = np.zeros((window,width_plate,height_plate))
    # print(time.time()-t, end="\r")
    # mass_mat = np.ones((width_plate,height_plate))
    ## mass_mat[ 0, :] = 0.01
    # mass_mat[-1, :] = 0.01
    ## mass_mat[ :, 0] = 0.01
    # mass_mat[ :,-1] = 0.01
    # mass_mat *= m

    for i in range(int(window)-1):
        posit[fixed_x, fixed_y] = 0
        posit[origin_x, origin_y] = np.mean(audio[i*time_res:(i+1)*time_res]) * vol
        # posit[0, 0] += audio[i*time_res] * vol
        # print(np.max(posit_acum))
        # print(np.min(posit_acum))
        # print("")

        # force = k * ( np.roll(posit, 1,axis=0)+
        #                         np.roll(posit,-1,axis=0)+
        #                         np.roll(posit, 1,axis=1)+
        #                         np.roll(posit,-1,axis=1)-
        #                                         4*posit)

        pos_sum =4*posit
        pos_sum[:-1,:] -= posit[1: ,:]*f_mul_1
        pos_sum[1: ,:] -= posit[:-1,:]
        pos_sum[:,:-1] -= posit[:, 1:]*f_mul_2
        pos_sum[:,1: ] -= posit[:,:-1]

        ## ADDING THE EFFECT OF DIAGONAL NEIGHBOURS
        # pos_sum[:-1,:-1] -= posit[1: ,1:]*f_mul_3*np.sqrt(2)/2
        # pos_sum[:-1, 1:] -= posit[1:,:-1]*f_mul_1[:,:-1]*np.sqrt(2)/2
        # pos_sum[1: ,1: ] -= posit[:-1,:-1]*np.sqrt(2)/2
        # pos_sum[1: ,:-1] -= posit[:-1,1:]*f_mul_2[:-1,:]*np.sqrt(2)/2
        force = k * (-pos_sum)

        # force += k*(np.sqrt(2)/2)*(np.roll(np.roll(posit, 1,axis=0), 1,axis=1)+
        #                                      np.roll(np.roll(posit, 1,axis=0),-1,axis=1)+
        #                                      np.roll(np.roll(posit,-1,axis=0), 1,axis=1)+
        #                                      np.roll(np.roll(posit,-1,axis=0),-1,axis=1)-
        #                                                                         4*posit)

        # accel = force/mass_mat
        accel = force/m
        veloc += accel*dt
        posit += veloc*dt
        posit_acum[i, :, :] = posit * pos_multiplier

    deviation = np.var(posit_acum[window//2-1:window-1, :, :], axis= 0)
    # deviation = np.max(posit_acum[window//2-1:window-1, :, :], axis= 0) - np.min(posit_acum[window//2-1:window-1, :, :], axis= 0)

    # print(np.max(posit_acum[window//2-1:window-1, 48, 48]))
    # print(np.min(posit_acum[window//2-1:window-1, 48, 48]))
    # print(np.max(posit_acum[window//2-1:window-1, 0, 0]))
    # print(np.min(posit_acum[window//2-1:window-1, 0, 0]))
    # print(force[48, 48])
    # print(force[1, 1])
    # print(deviation[48, 48])
    # print(deviation[1, 1])

    if upscale > 1:
        deviation_scaled = resize_matrix_lerp(deviation, res_width//2, res_height//2)
    else:
        deviation_scaled = deviation
    qframe = deviation_scaled > (threshold/100)
    # frame = deviation.astype(np.uint8)

    dwframe = np.concat((np.fliplr(qframe), qframe), axis = 1)

    frame = np.concat((np.flipud(dwframe), dwframe), axis = 0)

    # print("fps", 1/((time.time()-t)), end="\r")
    return (apply_thickness(frame, thickness))

def resize_matrix_lerp(matrix, new_rows, new_cols):
    old_rows, old_cols = matrix.shape

    x_old = np.linspace(0, 1, old_cols)
    x_new = np.linspace(0, 1, new_cols)
    y_old = np.linspace(0, 1, old_rows)
    y_new = np.linspace(0, 1, new_rows)

    temp = np.array([np.interp(x_new, x_old, row) for row in matrix])
    resized = np.array([np.interp(y_new, y_old, temp[:, col]) for col in range(new_cols)])
    return resized
