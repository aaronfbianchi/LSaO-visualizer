import typing
from typing import Annotated

import numpy as np

from util import FrameSequence, MonoSamples, StereoSamples, VideoFrame

from . import validators
from .common import (
    MonoVisualizer,
    SquareVisualizer,
    StereoVisualizer,
    Visualizer,
    VizParam,
    apply_thickness,
    cubic_hermite,
    note_to_frequency,
    split_into_frames,
)

type PlotStyle = typing.Literal["Points", "Curve", "Solid"]

def abs_slice(a: int, b: int) -> slice[int]:
    return slice(min(a, b), max(a, b))

def condense_domain[T: np.number](audio: MonoSamples[T], samples_per_pixel: float) -> np.ndarray[tuple[int, int], np.dtype[T]]:
    if samples_per_pixel < 1:
        num_pixels = round(len(audio) / samples_per_pixel)
        resampled = np.interp(np.linspace(0, len(audio) - 1, num_pixels), np.arange(len(audio)), audio)
        return resampled.astype(audio.dtype)[:, np.newaxis] # type: ignore
    elif samples_per_pixel == 1:
        return audio[:, np.newaxis]

    # For each audio sample, the location it will map to in the pixel-scaled downsampling.
    # e.g. for ratio 1920÷3000:
    # 0 1 1 2 3 3 4 4 5 6 6 7 8 8 9 10 10 11 12 12 13 13 14 ...
    labels = np.round(np.arange(len(audio)) / samples_per_pixel).astype(np.uint32)

    # Mask of label "boundaries". One at the start of every group, plus an extra 1 at the very end of the array.
    markers = np.diff(labels, prepend=-1, append=labels[-1]+1).astype(np.bool)
    range_starts = np.flatnonzero(markers[:-1])
    range_ends = np.flatnonzero(markers[1:])

    # The number of samples in each group.
    # Values will be a mixture of floor(1/ratio) and ceil(1/ratio).
    # (Except for the start and end, which will be half as large.)
    # e.g. for ratio 720÷50000:
    # 34 69 68 69 68 68 69 68 69 68 ...
    range_lengths = range_ends - range_starts + 1
    longest_range = np.arange(np.max(range_lengths))

    # Broadcast the "prototype" range, clamping to each range's size.
    # Effectively [np.arange(n) for n in range_sizes], but with rows padded to fill a rectangular array.
    ranges = np.fmin(longest_range[np.newaxis, :], range_lengths[:, np.newaxis] - 1)
    indices = range_starts[:, np.newaxis] + ranges

    return audio[indices]

class LinePlotVisualizer(Visualizer):
    style: Annotated[PlotStyle, VizParam("Drawing Style", order=1)]

    plot_centered: typing.ClassVar[bool] = False
    plot_transposed: typing.ClassVar[bool] = False

    def remap_to_video[S: tuple[int, ...]](
        self,
        data: np.ndarray[S, np.dtype[np.floating]],
        *,
        res: tuple[int, int] | None = None,
        mode: typing.Literal["normalize", "clip", "keep"] = "normalize"
    ) -> np.ndarray[S, np.dtype[np.int16]]:
        width, height = res or self.resolution
        size = width if self.plot_transposed else height

        if self.plot_centered:
            match mode:
                case "normalize": data /= np.max(np.abs(data))
                case "clip": np.clip(data, -1, 1, out=data)
                case "keep": pass
            data *= 0.5
            data *= 0.95
            data += 0.5
        else:
            match mode:
                case "normalize": data /= np.max(data) / 0.95
                case "clip": np.clip(data, 0, 1, out=data)
                case "keep": pass

        data *= size - 1
        return np.round(data, out=data).astype(np.int16)

    def plot_1d(
        self,
        data: np.ndarray[tuple[int, ...], np.dtype[np.int16]],
        *,
        out: VideoFrame | None = None,
        res: tuple[int, int] | None = None,
        use_thickness: bool = False,
    ) -> VideoFrame:
        if out is None:
            width, height = res or self.resolution
            out = np.empty((height, width), dtype=np.bool)

        if data.ndim == 1:
            data = data[:, np.newaxis]
        else:
            assert data.ndim == 2

        view = np.flipud(out)
        if self.plot_transposed:
            view = view.T

        height, width = view.shape

        # TODO: Figure out why this is necessary (if it still is), and make it not necessary.
        data = data[:width]

        y_axis = np.arange(height).reshape((height, 1))
        match self.style:
            case "Points":
                view.fill(False)
                np.put_along_axis(view, data.T, values=True, axis=0)
            case "Curve":
                min_values = np.min(data, axis=1)
                max_values = np.max(data, axis=1)
                # When the graph is slanted up, stretch each max_value up towards the next min_value
                np.fmax(max_values[:-1], min_values[1:] - 1, out=max_values[:-1])
                # When the graph is slanted down, stretch each min_value down towards the next max_value
                np.fmin(min_values[:-1], max_values[1:] + 1, out=min_values[:-1])
                np.logical_and([min_values] <= y_axis, y_axis <= [max_values], out=view)
            case "Solid":
                if self.plot_centered:
                    min_values = np.min(data, axis=1)
                    max_values = np.max(data, axis=1)
                    mid = height // 2
                    # Stretch the bottom of the curve down to the center, and the top up
                    np.fmin(mid, min_values, out=min_values)
                    np.fmax(mid, max_values, out=max_values)
                    np.logical_and([min_values] <= y_axis, y_axis <= [max_values], out=view)
                else:
                    max_values = np.max(data, axis=1)
                    np.less_equal(y_axis, [max_values], out=view)

        return apply_thickness(out, self.thickness) if use_thickness else out

def trim_fft[S: tuple[int, ...], T: np.dtype[np.number]](
    fft: np.ndarray[S, T],
    frequency_limits: tuple[int, int],
    sample_rate: int,
    *,
    min_size: int | None = None,
) -> np.ndarray[S, T]:
    xlow, xhigh = frequency_limits
    xlow = np.max((xlow, 1))
    xhigh = np.min((xhigh, sample_rate / 2))
    if min_size is not None:
        xhigh = np.max((xhigh, xlow + min_size))

    frame_size = fft.shape[-1]

    xlimlow = int(np.ceil(frame_size * xlow / sample_rate * 2)) - 1
    xlimhigh = int(np.ceil(frame_size * xhigh / sample_rate * 2)) - 1
    return fft[..., xlimlow:xlimhigh] # type: ignore

def interpolate_1d[S: tuple[int, ...], T: np.dtype[np.number]](
    data: np.ndarray[S, T],
    size: int,
) -> np.ndarray[S, np.dtype[np.float32]]:
    if len(data) == size:
        return data.astype(np.float32)

    derivative = np.diff(data, axis=0)
    tangents = np.zeros_like(data, dtype=np.float32)
    tangents[1:] += derivative
    tangents[:-1] += derivative
    tangents[1:-1] *= 0.5

    sample_points = np.linspace(0, len(data) - 1, size)
    i0 = np.floor(sample_points).astype(np.uint16)
    i1 = np.ceil(sample_points).astype(np.uint16)

    interpolated = cubic_hermite(t=sample_points%1, p0=data[i0], p1=data[i1], m0=tangents[i0], m1 = tangents[i1])
    return np.fmax(interpolated, 0, out=interpolated) # ty: ignore

class BaseSpectrumVisualizer(Visualizer):
    smoothing: Annotated[int, VizParam(
        "Time Smoothing",
        description="Gives you a curve that doesn't move that violently. Whole number.",
        validator=validators.positive_integer,
        ignored_in_live_mode=True,
    )] = 1

    frequency_limits: Annotated[tuple[int, int], VizParam(
        "Frequency Limits",
        description="Lower and higher frequency limits in Hz, respectively.\nFrom 1Hz to half the sample rate of the audio.",
        divider="-",
        placeholder=("MIN", "MAX"),
        values=[1,20,300],
        values2=[600,1000,2000,5000,13000,20000],
        validator=validators.positive_integer_range,
    )] = (20, 13000)

    def trim_fft[S: tuple[int, ...], T: np.dtype[np.number]](
        self,
        fft: np.ndarray[S, T],
        sample_rate: int,
        *,
        min_size: int | None = None,
    ) -> np.ndarray[S, T]:
        return trim_fft(fft, self.frequency_limits, sample_rate, min_size=min_size)


class SpectrumVisualizer(MonoVisualizer, BaseSpectrumVisualizer, LinePlotVisualizer):
    title = "Linear Spectrum Visualizer by Aaron F. Bianchi"
    style: PlotStyle = "Solid"

    attenuation_steep: Annotated[float, VizParam(
        "Mid/High Boost",
        description="This will boost everything but the low end. You can enter a negative\nvalue for some crazy results, but going below -10 has no purpose.",
    )] = 0.5

    limt_junk: Annotated[bool, VizParam(
        "Expander",
        description="This will reduce the intensity of small amplitudes.",
    )] = True

    ## THE BIGGER THIS VALUE, THE BIGGER THE AMPLITUDES HAVE TO BE TO NOT BE REJECTED
    junk_threshold: Annotated[float, VizParam(
        "Expand Threshold",
        description="The bigger this value, the bigger amplitudes have to be to not\nbe reduced. Doesn't have to be a whole number.",
    )] = 2

    ## THIS WILL MAKE THE TRANSITION BETWEEN BEING REJECTED OR PASSED MORE ABRUPT
    threshold_steep: Annotated[float, VizParam(
        "Expand Steepness",
        description="This will make the transition between amplitudes being reduced\nor boosted more abrupt. Doesn't have to be a whole number.",
    )] = 10

    def attenuate_fft[S: tuple[int, ...], T: np.floating](self, fft: np.ndarray[S, np.dtype[T]], *, normalize: bool) -> np.ndarray[S, np.dtype[T]]:
        amount = self.attenuation_steep
        if amount == 0.0: # and amount >= -0.00001:
            return fft
        elif amount < 0.0:
            amount = 1.0 / max(amount, -10.0)

        if normalize:
            fft /= np.max(fft)

        x_ax = np.linspace(0, 20, fft.shape[-1])
        x_ax = 1 - np.e ** (-x_ax / amount)

        # Bass attenuation
        fft *= x_ax

        if normalize:
            fft /= np.max(np.abs(fft))

        return fft

    def reject_junk[S: tuple[int, ...], T: np.floating](self, fft: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[np.float64]]:
        exponent = self.junk_threshold - self.threshold_steep * fft
        no_junk = 1.0 / (1.0 + np.e ** exponent)
        # set min to 0
        return no_junk - np.min(no_junk)

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        # if fil:
            # N = 91
            # h = np.cos(np.linspace(0,2*np.pi,N)) - 1
            # h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
            # audio = np.convolve(audio, h, mode = 'same')

        size_frame = int(np.round(sample_rate*self.smoothing/self.framerate))
        audioShaped = split_into_frames(audio, size_frame)
        audioShaped *= np.hamming(size_frame)

        fsong = abs(np.fft.rfft(audioShaped, axis = 1))
        fsong_trim = self.trim_fft(fsong, sample_rate)
        fsong_res = interpolate_1d(fsong_trim.T, self.resolution[0]).T
        if self.smoothing != 1:
            fsong_res = interpolate_1d(fsong_res, self.smoothing*fsong_res.shape[0])
        fsong_comp = self.attenuate_fft(fsong_res, normalize=True)
        if self.limt_junk:
            fsong_comp = self.reject_junk(fsong_comp)

        fsong_comp = self.remap_to_video(fsong_comp)
        return FrameSequence(fsong_comp.shape[0], (self.plot_1d(frame) for frame in fsong_comp))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        audio *= np.hamming(len(audio))
        fsong = abs(np.fft.rfft(audio))

        fsong_trim = self.trim_fft(fsong, sample_rate=48000, min_size=100)
        fsong_res = interpolate_1d(fsong_trim.T, resolution[0]).T
        fsong_res /= 50
        fsong_comp = self.attenuate_fft(fsong_res, normalize=False)
        if self.limt_junk:
            fsong_comp = self.reject_junk(fsong_comp)

        fsong_comp = self.remap_to_video(fsong_comp, res=resolution, mode="clip")
        return self.plot_1d(fsong_comp, res=resolution)

class SpectrumdBVisualizer(MonoVisualizer, BaseSpectrumVisualizer, LinePlotVisualizer):
    title = "Linear Spectrum Visualizer (dB) by Aaron F. Bianchi"
    style: PlotStyle = "Solid"

    min_dB: Annotated[float, VizParam(
        "Spectrum Floor",
        description="Minimum value to display in dB. Less than 0dB",
        validator=validators.negative,
    )] = -80

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        size_frame = int(np.round(sample_rate*self.smoothing/self.framerate))
        audioShaped = split_into_frames(audio, size_frame)
        audioShaped *= np.hamming(size_frame)

        fsong = abs(np.fft.rfft(audioShaped, axis = 1))

        fsong_trim = self.trim_fft(fsong, sample_rate)
        fsong_res = interpolate_1d(fsong_trim.T, self.resolution[0]).T
        if self.smoothing != 1:
            fsong_res = interpolate_1d(fsong_res, self.smoothing*fsong_res.shape[0])

        ################# DECIBELES
        fsong_res_2 = fsong_res/np.max(fsong_res) #NORMALIZATION
        low_dB = self.min_dB
        high_dB = 6
        fsong_res_2 = 20*np.log10(fsong_res_2)
        fsong_res_2 = (fsong_res_2 - low_dB) / (high_dB - low_dB)
        #fsong_res_2 = np.clip(fsong_res_2, low_dB, None)

        fsong_comp = self.remap_to_video(fsong_res_2, mode="clip")
        return FrameSequence(fsong_comp.shape[0], (self.plot_1d(frame) for frame in fsong_comp))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        audio *= np.hamming(len(audio))
        fsong = abs(np.fft.rfft(audio))

        fsong_trim = self.trim_fft(fsong, sample_rate=48000, min_size=100)
        fsong_res = interpolate_1d(fsong_trim, resolution[0])
        fsong_res /= 50

        low_dB = self.min_dB
        high_dB = 6
        fsong_res = 20*np.log10(fsong_res)
        fsong_res = (fsong_res - low_dB) / (high_dB - low_dB)
        #fsong_res = np.clip(fsong_res, low_dB, None)

        fsong_comp = self.remap_to_video(fsong_res, res=resolution, mode="clip")
        return self.plot_1d(fsong_comp, res=resolution)

class SpecBalanceVisualizer(StereoVisualizer, BaseSpectrumVisualizer, LinePlotVisualizer, SquareVisualizer):
    title = "Linear Spectral Balance Visualizer by Aaron F. Bianchi"
    style: PlotStyle = "Curve"
    plot_centered = True
    plot_transposed = True

    def generate_video(self, audio: StereoSamples[np.float16], sample_rate: int) -> FrameSequence:
        size_frame = int(np.round(sample_rate*self.smoothing/self.framerate))
        audioShaped = split_into_frames(audio, size_frame)
        audioShaped *= np.hamming(size_frame)

        fsong = abs(np.fft.rfft(audioShaped, axis=2))
        fsong_trim = self.trim_fft(fsong, sample_rate)

        fsong_res = interpolate_1d(fsong_trim.T, self.resolution[1]).T
        if self.smoothing != 1:
            fsong_res = interpolate_1d(fsong_res.swapaxes(0, 1), self.smoothing*fsong_res.shape[1]).swapaxes(0, 1)

        fsong_res_2 = np.log10(fsong_res)
        fsong_res_2 = np.logaddexp(10, fsong_res_2) #soft clipping
        # proper normalization
        fsong_res_2 -= np.min(fsong_res_2)
        fsong_res_2 /= np.max(fsong_res_2)

        fsong_vert = self.remap_to_video(fsong_res_2[1] - fsong_res_2[0], mode="keep")
        return FrameSequence(fsong_vert.shape[0], (self.plot_1d(frame) for frame in fsong_vert))

    def live_preview(self, audio: StereoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        audio *= np.hamming(audio.shape[1])

        fsong = abs(np.fft.rfft(audio))
        fsong_trim = self.trim_fft(fsong, 48000, min_size=100)
        fsong_res = interpolate_1d(fsong_trim.T, resolution[1]).T

        fsong_res_2 = np.log10(fsong_res)
        fsong_res_2 = np.logaddexp(10, fsong_res_2) #soft clipping

        fsong_vert = (fsong_res_2[1] - fsong_res_2[0])*2000
        fsong_vert = self.remap_to_video(fsong_vert, res=resolution, mode="clip")
        return self.plot_1d(fsong_vert, res=resolution)

type HistogramStyle = typing.Literal["Flat", "Linear", "Spline"]

class HistogramVisualizer(MonoVisualizer, LinePlotVisualizer, SquareVisualizer):
    title = "Histogram Visualizer by Aaron F. Bianchi"
    style: PlotStyle = "Solid"

    size_frame: Annotated[int, VizParam(
        "Smoothing",
        description="Number of samples of the histogram. Whole number.\nIf it's too low, it will be calculated with the FPS and sample rate.",
        validator=validators.positive_integer,
        ignored_in_live_mode=True,
    )] = 1000

    bars: Annotated[int, VizParam(
        "Bars",
        description="Number of bars of the histogram. Whole number.",
        validator=validators.positive_integer,
    )] = 101

    sensitivity: Annotated[float, VizParam(
        "Sensitivity",
        description="Makes smaller values more visible. Non-negative value.",
        validator=validators.non_negative,
    )] = 0.1

    curve_style: Annotated[HistogramStyle, VizParam("Curve Style")] = "Flat"

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        audio = np.clip(audio, -32768, 32767)

        n_frames = int(np.ceil(len(audio)*self.framerate/sample_rate))
        speed = len(audio)/n_frames

        size_frame = self.size_frame
        size_frame = max(size_frame, int(self.framerate/sample_rate)) #CALCULATE THE MINIMUM FRAME SIZE IN CASE IT'S TOO SHORT
        # if size_frame < int(self.framerate/sample_rate): #CALCULATE THE MINIMUM FRAME SIZE IN CASE IT'S TOO SHORT
        #     size_frame = int(self.framerate/sample_rate)
        audio = np.pad(audio, (0, int(speed*(n_frames) - len(audio) + size_frame))) ## TO COMPLETE THE LAST FRAME

        res_width = self.resolution[0]
        bars = self.bars
        #audioShaped = np.zeros((n_frames,size_frame))
        hist = np.zeros((n_frames,bars))
        resampled_hist = np.zeros((n_frames,res_width))

        if self.curve_style == "Flat": #RESMAPLING BADLY THE HISTOGRAM
            #AQUI SOLO SE OBTIENEN LAS POSICIONES DEL VETOR ORIGINAL PARA USARLO 100000 VECES DESPUES EN EL FOR
            idx = np.linspace(0, bars, res_width, endpoint=False)
            idx = np.floor(idx).astype(int)
        elif self.curve_style == "Linear":
            old_x = np.linspace(0, bars - 1, bars)
            new_x = np.linspace(0, bars - 1, res_width)

        for i in range(n_frames):
            audioShaped = audio[int(i*speed) : int(i*speed) + size_frame]
            # Don't know why, but for certain inputs, going all the way to -32768 seems to break things.
            hist[i,:], bins = np.histogram(audioShaped, bins=bars, range=(-32744, 32767))

            if self.curve_style == "Flat":
                resampled_hist[i,:] = hist[i,idx]
            elif self.curve_style == "Linear":
                resampled_hist[i,:] = np.interp(new_x, old_x, hist[i,:])

        if self.curve_style == "Spline":
            resampled_hist = interpolate_1d(hist.T, res_width).T

        #fsong_comp = (resampled_hist + 1)**0.5 - 1
        if self.sensitivity > 0:
            fsong_comp = np.log(self.sensitivity*resampled_hist+1)
        else:
            fsong_comp = resampled_hist

        fsong_comp = self.remap_to_video(fsong_comp)
        return FrameSequence(n_frames, (self.plot_1d(frame) for frame in fsong_comp))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        bars = np.maximum(self.bars,1) # TODO: Visualizers shouldn't need to do things like this
        res_width = resolution[0]
        resampled_hist = np.zeros((1,res_width))

        if self.curve_style == "Flat": #RESMAPLING BADLY THE HISTOGRAM
            #AQUI SOLO SE OBTIENEN LAS POSICIONES DEL VETOR ORIGINAL PARA USARLO 100000 VECES DESPUES EN EL FOR
            idx = np.linspace(0, bars, res_width, endpoint=False)
            idx = np.floor(idx).astype(int)
        elif self.curve_style == "Linear":
            old_x = np.linspace(0, bars - 1, bars)
            new_x = np.linspace(0, bars - 1, res_width)

        hist, bins = np.histogram(audio, bins=bars, range=(-1, 1))

        if self.curve_style == "Flat":
            resampled_hist = hist[idx]
        elif self.curve_style == "Linear":
            resampled_hist = np.interp(new_x, old_x, hist)
        elif self.curve_style == "Spline":
            resampled_hist = interpolate_1d(hist, resolution[0])

        if self.sensitivity > 0:
            norm_factor = np.log(self.sensitivity*(48000//60)+1)
            fsong_comp = np.log(self.sensitivity*resampled_hist+1)
        else:
            norm_factor = 48000//60
            fsong_comp = resampled_hist

        fsong_comp /= norm_factor
        fsong_comp = self.remap_to_video(fsong_comp, res=resolution, mode="clip")
        return self.plot_1d(fsong_comp, res=resolution)

class WaveformVisualizer(MonoVisualizer, LinePlotVisualizer):
    title = "Short Waveform Visualizer by Aaron F. Bianchi"
    # Long waveform plot style is ignored in live mode? Wack.
    # YEAH IT IS XD
    style: PlotStyle = "Curve"
    plot_centered = True

    # TODO: The visualizer shouldn't be responsible for parsing tuning.
    # It should go somewhere in between the UI and the number crunching.
    note: Annotated[str, VizParam(
        "Tuning",
        # Oscilloscope?
        "Set a note to tune the oscilloscope to.\nYou can enter the name of a note or its fundamental frequency in Hz.",
        validator=validators.tuning_frequency,
        ignored_in_live_mode=True,
    )] = "C2"

    window_size: Annotated[int, VizParam(
        "Window Size",
        description="The smaller this is, the faster the waveform will move. Whole number.\nRecommended minimum is the width of the video.\nFor higher than ~20000 I recommend using the long waveform.",
        validator=validators.positive_integer,
        ignored_in_live_mode=True,
    )] = 3000

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        freq_tune = note_to_frequency(self.note) # unit: cycles per second

        duration = len(audio) / sample_rate # unit: seconds
        speed = sample_rate / freq_tune # unit: samples per cycle (samples/sec ÷ cycles/sec)

        # Problem: we have an overdetermined system of equations which is very likely to be inconsistent.
        # For the desired "strobe" effect from the tuning parameter, the waveform must advance by one cycle per frame.
        # For the video to remain in sync with the audio, the waveform must advance at a rate proportional to duration (in samples) and framerate.
        # Duration, framerate, and tuning frequency are all user-specified variables.
        #
        # Solution: Imagine a version of the output video where framerate = tuning frequency.
        # This video will (almost by definition) fully satisfy the strobe tuning constraint.
        # From there, achieve the user-specified framerate by duplicating or dropping frames.
        # This results in temporal nearest-neighbor resampling.
        # The judder that results from this is hopefully not a big deal because the whole point is to freeze the waveform in place.
        desired_frame_count = duration * freq_tune
        actual_frame_count = int(duration * self.framerate)
        frame_indices = np.linspace(0, desired_frame_count - 1, actual_frame_count)
        np.round(frame_indices, out=frame_indices)

        # Start the video with an empty waveform.
        # For the end, the slicing logic in the frame generation loop should mean
        # that we can add arbitrary amounts of padding to the end,
        # and the code will only take whatever amount it actually needs.
        audio = np.pad(audio, (self.window_size, int(np.ceil(speed)) + self.window_size))

        samples_per_pixel = self.window_size / self.resolution[0]
        remapped = self.remap_to_video(audio)
        grouped = condense_domain(remapped, samples_per_pixel)
        speed /= samples_per_pixel

        def gen() -> typing.Generator[VideoFrame]:
            prev_end = 0 # Keep track of "where we left off", and therefore how much output we can reuse
            buffer = np.zeros(self.resolution[::-1], dtype=np.bool)
            for fidx in frame_indices:
                start = round(fidx * speed)
                end = start + self.resolution[0]

                m0 = max(0, prev_end - start) # The amount of pixels that can be reused from the previous frame
                if m0 > 0:
                    buffer[:, :m0] = buffer[:, -m0:]
                    m0 -= 1 # Redraw one extra pixel to prevent gaps in curve mode

                self.plot_1d(grouped[start+m0:end], out=buffer[:, m0:], use_thickness=False)
                yield apply_thickness(buffer, self.thickness)
                prev_end = end

        return FrameSequence(actual_frame_count, gen())

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        remapped = self.remap_to_video(audio, res=resolution, mode="clip")
        grouped = condense_domain(remapped, len(remapped) / resolution[0])
        return self.plot_1d(grouped, res=resolution)

# Not currently used as a base class for short waveform.
# That one moves so fast that it's not reasonable to plot all at once,
# so it's not yet clear how much code it will share with envelope and long waveform.
class SlidingWindowLinePlotVisualizer(LinePlotVisualizer):
    pad_end: Annotated[bool, VizParam(
        "Pad End",
        description="Extend the audio with silence so that the video ends with a flat line.",
        ignored_in_live_mode=True,
    )] = False

    window_size: Annotated[int, VizParam(
        "Window Size",
        description="The smaller this is, the faster the plot will move. Whole number.",
        validator=validators.positive_integer,
        ignored_in_live_mode=True,
    )]

    # TODO: Option for non-animated output?
    def slice_into_frames(self, full_image: VideoFrame, num_frames: int, speed_px: float) -> typing.Generator[VideoFrame]:
        res_width = self.resolution[0]
        for i in range(num_frames):
            start = int(i * speed_px)
            end = start + res_width
            yield full_image[:, start:end]


class LongWaveformVisualizer(MonoVisualizer, SlidingWindowLinePlotVisualizer):
    title = "Long Waveform Visualizer by Aaron F. Bianchi"
    style: PlotStyle = "Curve"
    window_size = 400_000
    plot_centered = True

    def __init__(self) -> None:
        super().__init__()
        self.oldWaveLive = np.zeros((48000//60,2))

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        num_samples = audio.shape[-1]
        duration_seconds = num_samples / sample_rate
        num_frames = int(np.ceil(duration_seconds * self.framerate))

        res_width, res_height = self.resolution
        samples_per_pixel = self.window_size / res_width # 208⅓ on default settings
        samples_per_frame = num_samples / num_frames     # ~735   on default settings @ 44.1kHz, ~800   @ 48kHZ
        speed_px = samples_per_frame / samples_per_pixel # ~3.528 on default settings @ 44.1kHz, ~3.840 @ 48kHz

        # Start the video with a flat waveform
        padding = (self.window_size, 0)
        full_width = res_width + int(np.ceil(speed_px * num_frames))

        if self.pad_end:
            # Also end the video with a flat waveform
            padding = (self.window_size, self.window_size)
            full_width += res_width
            num_frames += int(np.ceil(res_width / speed_px))

        audio = np.pad(audio, padding)
        remapped = self.remap_to_video(audio)
        grouped = condense_domain(remapped, samples_per_pixel)
        full_waveform = self.plot_1d(grouped, res=(full_width, res_height))

        return FrameSequence(num_frames, self.slice_into_frames(full_waveform, num_frames, speed_px))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        audio = np.clip(audio, -1, 1)

        res_width, res_height = resolution
        frameData = np.zeros((res_height, res_width), dtype=np.bool)

        self.oldWaveLive = np.roll(self.oldWaveLive, -1, axis=0)
        self.oldWaveLive[-1,1] = np.max(audio)
        self.oldWaveLive[-1,0] = np.min(audio)

        fsong_comp = self.remap_to_video(self.oldWaveLive.copy(), res=resolution, mode="clip")

        window_size = fsong_comp.shape[0]
        xaxis = np.linspace(0, resolution[0] - 1,window_size)

        for m in range(window_size):
            frameData[int(fsong_comp[m,0])-1 : int(fsong_comp[m,1]), int(xaxis[m])] = 1

        frameData = apply_thickness(frameData, self.thickness)

        return frameData

class EnvelopeVisualizer(MonoVisualizer, SlidingWindowLinePlotVisualizer):
    title = "Envelope Visualizer by Aaron F. Bianchi"
    style: PlotStyle = "Curve"
    window_size = 50_000

    smoothing: Annotated[int, VizParam(
        "Smoothing",
        description="This makes the envelope smoother. Positive integer.",
        validator=validators.positive_integer,
    )] = 1000

    def __init__(self) -> None:
        super().__init__()
        self.oldEnvLive = np.zeros(48000//60)

    def generate_video(self, audio: MonoSamples[np.float16], sample_rate: int) -> FrameSequence:
        # The actual envelope calculation
        kernel = np.ones(self.smoothing) / self.smoothing
        audio = np.convolve(np.abs(audio, out=audio), kernel, mode='same')

        # Most of these calculations are mostly shared with long waveform
        num_samples = audio.shape[-1]
        duration_seconds = num_samples / sample_rate
        num_frames = int(np.ceil(duration_seconds * self.framerate))

        res_width, res_height = self.resolution
        samples_per_pixel = self.window_size / res_width
        samples_per_frame = num_samples / num_frames
        speed_px = samples_per_frame / samples_per_pixel

        # Start the video with a flat waveform
        padding = (self.window_size, 0)
        full_width = res_width + int(np.ceil(speed_px * num_frames))

        if self.pad_end:
            padding = (self.window_size, self.window_size)
            full_width += res_width
            num_frames += int(np.ceil(res_width / speed_px))

        audio = np.pad(audio, padding)
        remapped = self.remap_to_video(audio)
        grouped = condense_domain(remapped, self.window_size / self.resolution[0])
        full_envelope = self.plot_1d(grouped, res=(full_width, res_height))

        return FrameSequence(num_frames, self.slice_into_frames(full_envelope, num_frames, speed_px))

    def live_preview(self, audio: MonoSamples[np.float16], resolution: tuple[int, int]) -> VideoFrame:
        self.oldEnvLive = np.roll(self.oldEnvLive, -1)
        self.oldEnvLive[-1] = np.max(np.abs(audio))
        self.oldEnvLive[-1] = self.oldEnvLive[-1]/(1+self.smoothing/1000) + self.oldEnvLive[-2]*(1-1/(1+self.smoothing/1000))

        remapped = self.remap_to_video(self.oldEnvLive.copy(), res=resolution, mode="clip")
        ratio = len(remapped) / resolution[0]
        if ratio > 1:
            grouped = condense_domain(remapped, len(remapped) / resolution[0])
        else:
            # Unlike most other modes, we want nearest-neighbor upsampling here,
            # because interpolation results in an undesirable "crawling" effect.
            # The "shimmer" from nearest-neighbor is a little bit less jarring.
            xaxis = np.linspace(0, len(remapped) - 1, resolution[0])
            indices = np.round(xaxis).astype(np.uint16)
            grouped = remapped[indices, np.newaxis]
        return self.plot_1d(grouped, res=resolution)
