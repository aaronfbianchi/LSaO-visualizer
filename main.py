#python3 -m venv myenv
#source myenv/bin/activate
#venv\Scripts\activate
#pip install scipy
#pip install pyinstaller
#pyinstaller --onefile --console main.py

import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
from scipy import signal
import subprocess
import os
#from scipy.signal import butter, filtfilt#, sosfreqz, sosfreqresp, sos2tf, sosfilter
import webbrowser
from PIL import Image, ImageTk
import shutil
import sys

######################################################################
## LINEAR SPRECTRUM AND OSCILLOSCOPE VISUALIZER by Aarón F. Bianchi ##
######################################################################

def read_audio_samples(input_file):
    cmd = [FFMPEG, '-i', input_file, '-f', 's16le', '-']
    output = subprocess.check_output(cmd, stderr=subprocess.PIPE)

    cmd_probe = [FFPROBE, '-show_streams', '-select_streams', 'a:0', input_file]
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

def convert_vid(input_audio, output_name, vidfor):
    if vidfor == ".gif":
        ffmpeg_command = [
            FFMPEG,
            '-i', "resources/temporary_file.mp4", 
            output_name, 
            '-y'
        ]
    elif vidfor == ".webp":
        ffmpeg_command = [
            FFMPEG,
            '-i', "resources/temporary_file.mp4",
            '-loop','0',
            output_name,
            '-y']
    elif vidfor == ".webm":
        ffmpeg_command = [
            FFMPEG,
            '-i', input_audio,
            '-i', "resources/temporary_file.mp4",
            '-c:v', 'libvpx-vp9',
            '-c:a', 'libopus', '-b:a', '320k',
            output_name,
            '-y'
        ]
    else:
        ffmpeg_command = [
            FFMPEG,
            '-i', input_audio,
            '-i', "resources/temporary_file.mp4",
            '-c:v', 'copy', '-strict', 'experimental', '-b:a', '320k',
            output_name,
            '-y'
        ]

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def ffmpeg_ubicacion():
    ffmpeg_PATH = shutil.which("ffmpeg")
    if ffmpeg_PATH:
        return ffmpeg_PATH

    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
        exe_dir = os.path.dirname(sys.executable)
    else: # Running from source
        base_path = os.path.dirname(os.path.abspath(__file__))
        exe_dir = base_path

    search_paths = []

    if sys.platform == "win32":
        search_paths += [
            os.path.join(base_path, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(exe_dir, "ffmpeg", "bin", "ffmpeg.exe"),
        ]
    else:
        search_paths += [
            os.path.join(base_path, "ffmpeg", "bin", "ffmpeg"),
            os.path.join(exe_dir, "ffmpeg", "bin", "ffmpeg"),
        ]

    for candidate in search_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("FFmpeg not found in PATH or alongside the app.")

def ffprobe_ubicacion():
    ffprobe_PATH = shutil.which("ffprobe")
    if ffprobe_PATH:
        return ffprobe_PATH

    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
        exe_dir = os.path.dirname(sys.executable)
    else: # Running from source
        base_path = os.path.dirname(os.path.abspath(__file__))
        exe_dir = base_path

    search_paths = []

    if sys.platform == "win32":
        search_paths += [
            os.path.join(base_path, "ffmpeg", "bin", "ffprobe.exe"),
            os.path.join(exe_dir, "ffmpeg", "bin", "ffprobe.exe"),
        ]
    else:
        search_paths += [
            os.path.join(base_path, "ffmpeg", "bin", "ffprobe"),
            os.path.join(exe_dir, "ffmpeg", "bin", "ffprobe"),
        ]

    for candidate in search_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("FFprobe not found in PATH or alongside the app.")

def generate_spectrum(output_name,input_audio, channel,fps, res_width, res_height, t_smoothing, xlow, xhigh, limt_junk, attenuation_steep, junk_threshold, threshold_steep, style, thickness, compression, callback_function):
    root, vidfor = os.path.splitext(output_name)

    print("poto")
    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    #oversampling = 8 ## SMOOTHING IN THE CURVE (INTEGER)
    #t_smoothing = 1 ## TIME SMOOTHING (INTEGER)
    print(f"song {song.shape}")
    print(f"song {song.shape[0]}")
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            audio = np.transpose(np.mean(song, axis = 0))
            print(f"audio {audio.shape}")
        elif channel == "Left":
            audio = np.transpose(song[0,:])
        elif channel == "Right":
            audio = np.transpose(song[1,:])
    else:
        audio = np.transpose(song)
        print(f"audio {audio.shape}")

    # if fil:
        # N = 91
        # h = np.cos(np.linspace(0,2*np.pi,N)) - 1
        # h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
        # audio = np.convolve(audio, h, mode = 'same')
    
    size_frame = int(np.round(fs*t_smoothing/fps))
    n_frames = int(np.ceil(len(audio)/size_frame))
    
    audio = np.pad(audio, (0, int(size_frame*n_frames) - len(audio))) ## TO COMPLETE THE LAST FRAME
    
    audioShaped = np.zeros((n_frames,size_frame))
    for i in range(n_frames):
        audioShaped[i,:] = audio[i*size_frame : (i+1)*size_frame]
    
    w_hamming = np.zeros(size_frame)
    for h in range(size_frame): ## WINDOWING FOR CLEANER CURVE
        w_hamming[h] = 0.54 - 0.46*np.cos(2*np.pi*(h+1)/size_frame)
    
    for i in range(n_frames):
        audioShaped[i,:] = audioShaped[i,:]*w_hamming
    
    fsong = abs(np.fft.rfft(audioShaped, axis = 1))
    
    #xlow = 1 ## LOWER LIMIT FREQ. TO BE DISPLAYED
    #xhigh = 13000 ## HIGHER LIMIT FREQ. TO BE DISPLAYED
    xlow = np.max((xlow,1))
    xhigh = np.min((xhigh,fs/2))
    xlimlow = int(np.ceil(fsong.shape[1]*xlow/fs*2)) - 1
    xlimhigh = int(np.ceil(fsong.shape[1]*xhigh/fs*2)) - 1
    
    fsong_trim = fsong[:,xlimlow:xlimhigh]
    extra_width = 0.25 ##EXTRA WIDTH TO HOUSE THE WIERD RESAMPLE ARTIFACTS THAT ARE LATER GONNA BE DELETED
    fsong_trim_pad = np.concatenate((fsong_trim, np.zeros((fsong_trim.shape[0],int(fsong_trim.shape[1]*extra_width)))), axis = 1)
    #fsong_res = abs(signal.resample(fsong_trim, res_width, axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsong_res = abs(signal.resample(fsong_trim_pad, int(res_width*(1 + extra_width)), axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsong_res = fsong_res[:, 0:int(fsong_res.shape[1]/(1 + extra_width))]
    if (t_smoothing == 1):
        fsong_res_2 = fsong_res
    else:
        fsong_res_2 = abs(signal.resample(fsong_res, t_smoothing*fsong_res.shape[0]))

    
    if attenuation_steep == 0.0: # and attenuation_steep >= -0.00001:
        fsong_comp = fsong_res_2 ## NO ATTENUATION
    else: ## BASS ATTENUATION
        if attenuation_steep < 0.0:
            attenuation_steep = np.max((attenuation_steep,-10))
            attenuation_steep = 1/attenuation_steep

        fsong_comp = fsong_res_2/np.max(fsong_res_2) ## NORMALIZATION
        x_ax = np.linspace(0,20,fsong_comp.shape[1])
        #x_ax = 1/(1 + np.e**(bass_attenuation - attenuation_steep*x_ax))
        #x_ax = x_ax - np.min(x_ax)
        x_ax = 1 - np.e**(-x_ax/(attenuation_steep))
        for j in range(fsong_comp.shape[0]):
            fsong_comp[j,:] = fsong_comp[j,:]*x_ax ## BASS ATTENUATION
        fsong_comp = abs(fsong_comp)/np.max(abs(fsong_comp)) ## NORMALIZATION
    
    if limt_junk:
        #junk_threshold = 3 ## THE BIGGER THIS VALUE, THE BIGGER THE AMPLITUDES HAVE TO BE TO NOT BE REJECTED
        #threshold_steep = 10 ## THIS WILL MAKE THE TRANSITION BETWEEN BEING REJECTED OR PASSED MORE ABRUPT
    
        fsong_comp = fsong_comp/np.max(fsong_comp) ## NORMALIZATION
        fsong_comp = 1/(1 + np.e**(junk_threshold - threshold_steep*fsong_comp)) ## JUNK REJECTION AND LIMITING
        fsong_comp = fsong_comp - np.min(fsong_comp) ## SET MIN TO 0
    
    fsong_comp = 0.95*res_height*(fsong_comp/np.max(fsong_comp)) ## NORMALIZATION AGAIN
    
    #style = 2 ## STYLE OF THE DRAWING
    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Spectrum": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True
    
    # Number of frames in the video
    num_frames = fsong_comp.shape[0]
    
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(num_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)
        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(res_width):
                    frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
    
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(res_width - 1):
                    point1 = fsong_comp[i,m]
                    point2 = fsong_comp[i,m+1]
                    if  int(point1) == int(point2):
                        frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
                    if  int(point1) > int(point2):
                        frameData[res_height - int(point1) -1: res_height - int(point2) -1, m] = True
                    else:
                        frameData[res_height - int(point2) -1: res_height - int(point1) -1, m] = True
        else: ## FILLED SPECTRUM
            for m in range(res_width):
                frameData[(res_height - int(fsong_comp[i,m])):res_height, m] = True
    
    
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{num_frames}")
        callback_function(i,num_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_spectrum_dB(output_name,input_audio, channel,fps, res_width, res_height, t_smoothing, xlow, xhigh, min_dB, style, thickness, compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    #oversampling = 8 ## SMOOTHING IN THE CURVE (INTEGER)
    #t_smoothing = 1 ## TIME SMOOTHING (INTEGER)
    print(f"song {song.shape}")
    print(f"song {song.shape[0]}")
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            audio = np.transpose(np.mean(song, axis = 0))
            print(f"audio {audio.shape}")
        elif channel == "Left":
            audio = np.transpose(song[0,:])
        elif channel == "Right":
            audio = np.transpose(song[1,:])
    else:
        audio = np.transpose(song)
        print(f"audio {audio.shape}")

    # if fil:
    #    N = 91
    #    h = np.cos(np.linspace(0,2*np.pi,N)) - 1
    #    h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
    #    audio = np.convolve(audio, h, mode = 'same')
    
    size_frame = int(np.round(fs*t_smoothing/fps))
    n_frames = int(np.ceil(len(audio)/size_frame))
    
    audio = np.pad(audio, (0, int(size_frame*n_frames) - len(audio))) ## TO COMPLETE THE LAST FRAME
    
    audioShaped = np.zeros((n_frames,size_frame))
    for i in range(n_frames):
        audioShaped[i,:] = audio[i*size_frame : (i+1)*size_frame]
    
    w_hamming = np.zeros(size_frame)
    for h in range(size_frame): ## WINDOWING FOR CLEANER CURVE
        w_hamming[h] = 0.54 - 0.46*np.cos(2*np.pi*(h+1)/size_frame)
    
    for i in range(n_frames):
        audioShaped[i,:] = audioShaped[i,:]*w_hamming
    
    fsong = abs(np.fft.rfft(audioShaped, axis = 1))
    
    #xlow = 1 ## LOWER LIMIT FREQ. TO BE DISPLAYED
    #xhigh = 13000 ## HIGHER LIMIT FREQ. TO BE DISPLAYED
    xlow = np.max((xlow,1))
    xhigh = np.min((xhigh,fs/2))
    xlimlow = int(np.ceil(fsong.shape[1]*xlow/fs*2)) - 1
    xlimhigh = int(np.ceil(fsong.shape[1]*xhigh/fs*2)) - 1
    
    fsong_trim = fsong[:,xlimlow:xlimhigh]
    extra_width = 0.25 ##EXTRA WIDTH TO HOUSE THE WIERD RESAMPLE ARTIFACTS THAT ARE LATER GONNA BE DELETED
    fsong_trim_pad = np.concatenate((fsong_trim, np.zeros((fsong_trim.shape[0],int(fsong_trim.shape[1]*extra_width)))), axis = 1)
    #fsong_res = abs(signal.resample(fsong_trim, res_width, axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsong_res = abs(signal.resample(fsong_trim_pad, int(res_width*(1 + extra_width)), axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsong_res = fsong_res[:, 0:int(fsong_res.shape[1]/(1 + extra_width))]
    if (t_smoothing == 1):
        fsong_res_2 = fsong_res
    else:
        fsong_res_2 = abs(signal.resample(fsong_res, t_smoothing*fsong_res.shape[0]))

    ################# DECIBELES
    fsong_res_2 = fsong_res_2/np.max(fsong_res_2) #NORMALIZATION
    low_dB = min_dB
    high_dB = 6
    #linear_min = 10 ** (low_dB / 20)
    #linear_max = 10 ** (high_dB / 20)
    #fsong_res_2 = fsong_res_2 * (linear_max - linear_min) + linear_min
    fsong_res_2 = 20*np.log10(fsong_res_2)
    fsong_res_2 = (fsong_res_2 - low_dB) / (high_dB - low_dB)
    fsong_res_2 = np.clip(fsong_res_2, low_dB, None) #SET FLOOR DB
    fsong_res_2 = fsong_res_2/np.max(fsong_res_2) #NORMALIZATION
    fsong_res_2 = np.clip(fsong_res_2, 0, 1) # CLIP FROM 0 TO 1
    
    fsong_comp = 0.95*res_height*(fsong_res_2/np.max(fsong_res_2)) ## NORMALIZATION AGAIN
    
    #style = 2 ## STYLE OF THE DRAWING
    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Spectrum": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True
    
    # Number of frames in the video
    num_frames = fsong_comp.shape[0]
    
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(num_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)
        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(res_width):
                    frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
    
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(res_width - 1):
                    point1 = fsong_comp[i,m]
                    point2 = fsong_comp[i,m+1]
                    if  int(point1) == int(point2):
                        frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
                    if  int(point1) > int(point2):
                        frameData[res_height - int(point1) -1: res_height - int(point2) -1, m] = True
                    else:
                        frameData[res_height - int(point2) -1: res_height - int(point1) -1, m] = True
        else: ## FILLED SPECTRUM
            for m in range(res_width):
                frameData[(res_height - int(fsong_comp[i,m])):res_height, m] = True
    
    
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{num_frames}")
        callback_function(i,num_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
    
def generate_spec_balance(output_name,input_audio,fps, res_width, res_height, t_smoothing, xlow, xhigh, style, thickness, compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    #oversampling = 8 ## SMOOTHING IN THE CURVE (INTEGER)
    #t_smoothing = 1 ## TIME SMOOTHING (INTEGER)
    print(f"song {song.shape}")
    print(f"song {song.shape[0]}")
    
    audioL = np.transpose(song[0,:])
    audioR = np.transpose(song[1,:])
    
    size_frame = int(np.round(fs*t_smoothing/fps))
    n_frames = int(np.ceil(len(audioL)/size_frame))
    
    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    
    audioLShaped = np.zeros((n_frames,size_frame))
    audioRShaped = np.zeros((n_frames,size_frame))
    for i in range(n_frames):
        audioLShaped[i,:] = audioL[i*size_frame : (i+1)*size_frame]
        audioRShaped[i,:] = audioR[i*size_frame : (i+1)*size_frame]

    w_hamming = np.zeros(size_frame)
    for h in range(size_frame): ## WINDOWING FOR CLEANER CURVE
        w_hamming[h] = 0.54 - 0.46*np.cos(2*np.pi*(h+1)/size_frame)
    for i in range(n_frames):
        audioLShaped[i,:] = audioLShaped[i,:]*w_hamming
        audioRShaped[i,:] = audioRShaped[i,:]*w_hamming

    fsongL = abs(np.fft.rfft(audioLShaped, axis = 1))
    fsongR = abs(np.fft.rfft(audioRShaped, axis = 1))
    
    #xlow = 1 ## LOWER LIMIT FREQ. TO BE DISPLAYED
    #xhigh = 13000 ## HIGHER LIMIT FREQ. TO BE DISPLAYED
    xlow = np.max((xlow,1))
    xhigh = np.min((xhigh,fs/2))
    xlimlow = int(np.ceil(fsongL.shape[1]*xlow/fs*2)) - 1
    xlimhigh = int(np.ceil(fsongL.shape[1]*xhigh/fs*2)) - 1
    
    fsongL_trim = fsongL[:,xlimlow:xlimhigh]
    fsongR_trim = fsongR[:,xlimlow:xlimhigh]
    extra_width = 0.25 ##EXTRA WIDTH TO HOUSE THE WIERD RESAMPLE ARTIFACTS THAT ARE LATER GONNA BE DELETED
    fsongL_trim_pad = np.concatenate((fsongL_trim, np.zeros((fsongL_trim.shape[0],int(fsongL_trim.shape[1]*extra_width)))), axis = 1)
    fsongR_trim_pad = np.concatenate((fsongR_trim, np.zeros((fsongR_trim.shape[0],int(fsongR_trim.shape[1]*extra_width)))), axis = 1)
    #fsong_res = abs(signal.resample(fsong_trim, res_width, axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsongL_res = abs(signal.resample(fsongL_trim_pad, int(res_height*(1 + extra_width)), axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsongR_res = abs(signal.resample(fsongR_trim_pad, int(res_height*(1 + extra_width)), axis = 1)) ## RESAMPLING TO THE WIDTH OF THE VIDEO
    fsongL_res = fsongL_res[:, 0:int(fsongL_res.shape[1]/(1 + extra_width))]
    fsongR_res = fsongR_res[:, 0:int(fsongR_res.shape[1]/(1 + extra_width))]
    if (t_smoothing == 1):
        fsongL_res_2 = fsongL_res
        fsongR_res_2 = fsongR_res
    else:
        fsongL_res_2 = abs(signal.resample(fsongL_res, t_smoothing*fsongL_res.shape[0]))
        fsongR_res_2 = abs(signal.resample(fsongR_res, t_smoothing*fsongR_res.shape[0]))
        
    fsongL_res_2 = np.log10(fsongL_res_2)
    fsongR_res_2 = np.log10(fsongR_res_2)
    print(f"fsongL_res_2 {fsongL_res_2}")
    print(f"fsongR_res_2 {fsongR_res_2}")
    #fsongL_res_2 = np.clip(fsongL_res_2, 3, None) #clipping
    #fsongR_res_2 = np.clip(fsongR_res_2, 3, None) #clipping
    fsongL_res_2 = np.log(np.e**10 + np.e**fsongL_res_2) #soft clipping
    fsongR_res_2 = np.log(np.e**10 + np.e**fsongR_res_2) #soft clipping
    print(f"fsongL_res_2 {fsongL_res_2}")
    print(f"fsongR_res_2 {fsongR_res_2}")
    gmin = np.min([np.min(fsongL_res_2), np.min(fsongR_res_2)])
    gmax = np.max([np.max(fsongL_res_2), np.max(fsongR_res_2)])
    fsongL_res_2 = (fsongL_res_2 - gmin)/(gmax - gmin) # proper normalization
    fsongR_res_2 = (fsongR_res_2 - gmin)/(gmax - gmin) # proper normalization
    print(" a")
    print(f"np.min(fsongL_res_2) {np.min(fsongL_res_2)}")
    print(f"np.min(fsongR_res_2) {np.min(fsongR_res_2)}")
    print(f"np.max(fsongL_res_2) {np.max(fsongL_res_2)}")
    print(f"np.max(fsongR_res_2) {np.max(fsongR_res_2)}")
    #fsongL_res_2 = fsongL_res_2/np.max([np.max(fsongL_res_2),np.max(fsongR_res_2)]) #NORMALIZING WITH BOTH CHANNELS
    #fsongR_res_2 = fsongR_res_2/np.max([np.max(fsongL_res_2),np.max(fsongR_res_2)]) #NORMALIZING WITH BOTH CHANNELS
    print(f"fsongL_res_2 {fsongL_res_2}")
    print(f"fsongR_res_2 {fsongR_res_2}")

    fsong_vert = fsongR_res_2 - fsongL_res_2
    #fsong_vert = np.clip(np.log10(fsong_vert), 0.000001, 100000)
    print(" b")
    fsong_vert = 0.95*res_width/2*(fsong_vert) + res_width/2
    print(fsong_vert)
    print(" c")
    #style = 2 ## STYLE OF THE DRAWING
    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Spectrum": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True
    
    # Number of frames in the video
    num_frames = fsong_vert.shape[0]
    
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    width_o2 = int(res_width/2)
    # Generate and save each frame as an image
    for i in range(num_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)
        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(res_height):
                    #print(" xd")
                    #print(m)
                    #print(res_height - m)
                    #print(int(fsong_vert[i,m]) - 1)
                    frameData[res_height - m - 1, int(fsong_vert[i,m]) - 1] = True
    
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(res_height - 1):
                    point1 = fsong_vert[i,m]
                    point2 = fsong_vert[i,m+1]
                    if  int(point1) == int(point2):
                        frameData[res_height - m - 1, int(point1) - 1] = True
                    if  point1 >= point2:
                        frameData[res_height - m - 1, int(point2) -1: int(point1) -1] = True
                    else:
                        frameData[res_height - m - 1, int(point1) -1: int(point2) -1] = True
        else: ## FILLED SPECTRUM
            print(" xv")
            for m in range(res_height):
                point1 = int(fsong_vert[i,m])
                if point1 < width_o2:
                    frameData[res_height - m - 1, point1- res_width:width_o2] = True
                else:
                    frameData[res_height - m - 1, width_o2:point1- res_width] = True
   
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{num_frames}")
        callback_function(i,num_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
 
def generate_histogram(output_name,input_audio, channel,fps, res_width, res_height, size_frame, bars, sensitivity, curve_style, style, thickness, compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    #oversampling = 8 ## SMOOTHING IN THE CURVE (INTEGER)
    #t_smoothing = 1 ## TIME SMOOTHING (INTEGER)
    print(f"song {song.shape}")
    print(f"song {song.shape[0]}")
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            audio = np.transpose(np.mean(song, axis = 0))
            print(f"audio {audio.shape}")
        elif channel == "Left":
            audio = np.transpose(song[0,:])
        elif channel == "Right":
            audio = np.transpose(song[1,:])
    else:
        audio = np.transpose(song)
        print(f"audio {audio.shape}")
    #print(np.max(audio))
    #print(np.min(audio))
    audio = np.clip(audio, -32768, 32767)
    
    n_frames = int(np.ceil(len(audio)*fps/fs))
    speed = len(audio)/n_frames
    
    if size_frame < int(fs/fps): #CALCULATE THE MINIMUM FRAME SIZE IN CASE IT'S TOO SHORT
        size_frame = int(fs/fps)
    audio = np.pad(audio, (0, int(speed*(n_frames) - len(audio) + size_frame))) ## TO COMPLETE THE LAST FRAME
    
    #audioShaped = np.zeros((n_frames,size_frame))
    hist = np.zeros((n_frames,bars))
    resampled_hist = np.zeros((n_frames,res_width))
    
    if curve_style == "Flat": #RESMAPLING BADLY THE HISTOGRAM
        repeat_counts = np.floor(res_width * np.ones(bars) / bars).astype(int)
        remainder = res_width - repeat_counts.sum()
        repeat_counts[:remainder] += 1
    elif curve_style == "Linear":
        old_x = np.linspace(0, bars - 1, bars)
        new_x = np.linspace(0, bars - 1, res_width)
        
    for i in range(n_frames):
        audioShaped = audio[int(i*speed) : int(i*speed) + size_frame]
        hist[i,:], bins = np.histogram(audioShaped, bins=bars, range=(-32768, 32767))
        
        if curve_style == "Flat":
            resampled_hist[i,:] = np.repeat(hist[i,:], repeat_counts)
            #resampled_hist[i,:] = resampled_hist[i,:]/np.max(resampled_hist[i,:])
        elif curve_style == "Linear":
            resampled_hist[i,:] = np.interp(new_x, old_x, hist[i,:])
    if curve_style == "FFT Resample":
        resampled_hist = abs(signal.resample(hist, res_width, axis = 1))
        
    #fsong_comp = (resampled_hist + 1)**0.5 - 1
    if sensitivity > 0:
        fsong_comp = np.log(sensitivity*resampled_hist+1)
    else:
        fsong_comp = resampled_hist
    fsong_comp = (res_height-1)*(fsong_comp/np.max(fsong_comp)).astype(np.float16) ## NORMALIZATION AGAIN
    
    #style = 2 ## STYLE OF THE DRAWING
    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Histogram": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True
    
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)
        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(res_width):
                    frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
    
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(res_width - 1):
                    point1 = fsong_comp[i,m]
                    point2 = fsong_comp[i,m+1]
                    #if  int(point1) == int(point2):
                    frameData[res_height - int(fsong_comp[i,m]) - 1, m] = True
                    if  int(point1) > int(point2):
                        frameData[res_height - int(point1) -1: res_height - int(point2) -1, m] = True
                    else:
                        frameData[res_height - int(point2) -1: res_height - int(point1) -1, m] = True
        else: ## FILLED SPECTRUM
            for m in range(res_width):
                frameData[(res_height - int(fsong_comp[i,m])):res_height, m] = True
    
    
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_waveform(output_name,input_audio,channel,fps_2, res_width, res_height, note, window_size, style,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    print(f"song {song.shape}")
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            audio = np.transpose(np.mean(song, axis = 0))
            print(f"audio {audio.shape}")
        elif channel == "Left":
            audio = np.transpose(song[0,:])
        elif channel == "Right":
            audio = np.transpose(song[1,:])
    else:
        audio = np.transpose(song)
        print(f"audio {audio.shape}")
        
    audio = audio/abs(np.max(audio)) #NORMALIZATION

    freq_tune = note_to_frequency(note)
    audio = np.concatenate((np.zeros((round(window_size))), audio)) ## TO ENSURE YOU HAVE A FIRST FRAME FULL OF 0's
    
    duration = len(audio)/fs

    speed = fs/freq_tune
    fps = fs/speed

    n_frames = int(np.ceil(duration*fps))
    n_frames_2 = round(duration*fps_2) #CANTIDAD FINAL DE FRAMES
    audio = np.pad(audio, (0, int(n_frames*speed + window_size) - len(audio)))
    #segments = np.zeros((n_frames + 1,window_size))

    indexes = (np.linspace(0, duration*fps - 1, int(duration*fps_2))) ## FPS ARE NOW 60
    indexes2 = [round(x) for x in indexes]

    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Waveform": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True

    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps_2),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    xaxis = np.linspace(0,res_width - 1,window_size)
    
    speed_px = int(speed*res_width/window_size)   
    oldFrame = np.zeros((res_height, res_width), dtype=bool)
    
    print(f"speed{speed}")
    print(f"speed_px{speed_px}")
    
    cont_ch = np.zeros(len(indexes2), dtype = bool)
    cont_ch[0] = 0
    for j in range(1, len(indexes2)): #CREATES A CHECK LIST FOR FRAMES THAT NEED TO BE REDRAWN COMPLETELY BECAUSE OF SKIPED FRAMES
        cont_ch[j] = np.max((0,2 - (indexes2[j] - indexes2[j-1]))).astype(bool)
    
    # Generate and save each frame as an image
    for i in range(n_frames_2-1):
        segments = audio[round(indexes2[i]*speed) : round(indexes2[i]*speed) + window_size]
        #seg_resa = np.clip(segments, -1, 1)
        fsong_comp = res_height/2 + segments*0.95*res_height/2

        frameData = np.roll(oldFrame, -speed_px ,axis = 1) #RECICLE THE LAST GENERATED FRAME
        frameData[:, res_width-speed_px:res_width] = False
        frameData = frameData*cont_ch[i]
        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(int(np.max((0,(window_size - speed))))*cont_ch[i],window_size):
                    frameData[res_height - int(fsong_comp[m]) -1, int(xaxis[m])] = True
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(int(np.max((0,(window_size - speed))))*cont_ch[i],window_size - 1):
                    point1 = int(fsong_comp[m])
                    point2 = int(fsong_comp[m+1])
                    if  point1 == point2:
                        frameData[res_height - point1 -1, int(xaxis[m])] = True
                    if  point1 > point2:
                        frameData[res_height - point1 -1: res_height - point2 -1, int(xaxis[m])] = True
                    else:
                        frameData[res_height - point2 -1: res_height - point1 -1, int(xaxis[m])] = True
        else: ## FILLED SPECTRUM
            for m in range(int(np.max((0,(window_size - speed))))*cont_ch[i],window_size):
                point1 = int(fsong_comp[m])
                if point1 < res_height/2:
                    frameData[int(res_height/2):res_height - point1, int(xaxis[m])] = True
                else:
                    frameData[res_height - point1:int(res_height/2), int(xaxis[m])] = True
                    
        oldFrame = frameData    
    
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
    
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames_2}")
        callback_function(i,n_frames_2, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames_2, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames_2, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_waveform_long(output_name,input_audio,channel,fps, res_width, res_height,window_size, style,thickness,compression, callback_function):
    
    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    
    print(f"song {song.shape}")
    if song.shape[0] == 2:
        if channel == "Both (Merge to mono)":
            audio = np.transpose(np.mean(song, axis = 0))
            print(f"audio {audio.shape}")
        elif channel == "Left":
            audio = np.transpose(song[0,:])
        elif channel == "Right":
            audio = np.transpose(song[1,:])
    else:
        audio = np.transpose(song)
        print(f"audio {audio.shape}")
        
    audio = audio/abs(np.max(audio)) #NORMALIZATION

    audio = np.concatenate((np.zeros((round(window_size))), audio)) ## TO ENSURE YOU HAVE A FIRST FRAME FULL OF 0's
    
    duration = len(audio)/fs #IN SECONDS
    speed = fs/fps #IN SAMPLES

    n_frames = int(np.ceil(duration*fps))
    audio = np.pad(audio, (0, int(n_frames*speed + window_size) - len(audio)))
    #segments = np.zeros((n_frames + 1,window_size))

    if style == "Just Points": ## DRAWS DOTS IN SCREEN
        points = True
        filled = False
    elif style == "Curve": ## DRAWS LINE IN SCREEN
        points = False
        filled = False
    elif style == "Filled Waveform": ## DRAWS FILLED SPECTRUM
        points = False
        filled = True

    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    xaxis = np.linspace(0,res_width - 1,window_size)
    
    speed_px = int(speed*res_width/window_size)   
    oldFrame = np.zeros((res_height, res_width), dtype=bool)
    
    print(f"speed{speed}")
    print(f"speed_px{speed_px}")
    
    # Generate and save each frame as an image
    for i in range(n_frames):
        segments = audio[round(i*speed) : round(i*speed) + window_size]
        #seg_resa = np.clip(segments, -1, 1)
        fsong_comp = res_height/2 + segments*0.95*res_height/2

        frameData = np.roll(oldFrame, -speed_px ,axis = 1) #RECICLE THE LAST GENERATED FRAME
        frameData[:, res_width-speed_px:res_width] = False

        if filled == False:
            if points:## DRAWS JUST POINTS
                for m in range(int(np.max((0,(window_size - speed)))),window_size):
                    frameData[res_height - int(fsong_comp[m]) -1, int(xaxis[m])] = True
            else: ## DRAWS A LINE (1.5x SLOW)
                for m in range(int(np.max((0,(window_size - speed)))),window_size - 1):
                    point1 = int(fsong_comp[m])
                    point2 = int(fsong_comp[m+1])
                    if  point1 == point2:
                        frameData[res_height - point1 -1, int(xaxis[m])] = True
                    if  point1 > point2:
                        frameData[res_height - point1 -1: res_height - point2 -1, int(xaxis[m])] = True
                    else:
                        frameData[res_height - point2 -1: res_height - point1 -1, int(xaxis[m])] = True
        else: ## FILLED SPECTRUM
            for m in range(int(np.max((0,(window_size - speed)))),window_size):
                point1 = int(fsong_comp[m])
                if point1 < res_height/2:
                    frameData[int(res_height/2):res_height - point1, int(xaxis[m])] = True
                else:
                    frameData[res_height - point1:int(res_height/2), int(xaxis[m])] = True
                    
        oldFrame = frameData    
    
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
    
        ffmpeg_process.stdin.write(frameData)
        #print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_oscilloscope(output_name,input_audio,fps, res_width, res_height,interpolation,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    
    print(f"shape song {song.shape}")
    
    song = song/np.max(np.max(abs(abs(song)))).astype(np.float16) ##TRANSPOSITION AND NORMALIZATION
    song = np.clip(song,-1,1)
    
    audioL = song[:,0].astype(np.float16)
    audioR = -song[:,1].astype(np.float16)
    
    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(len(audioL)/size_frame))
     
    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    print(f"shape audioL {audioL.shape}")
    
    extra_margin = 50
    audioL = np.pad(audioL,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audioR = np.pad(audioR,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    print(f"shape audioL {audioL.shape}")
    print(" ")
    audioLShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audioRShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(" ")
    for i in range(n_frames):
        audioLShaped[i,:] = audioL[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audioRShaped[i,:] = audioR[i*size_frame : (i+1)*size_frame + extra_margin*2]
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(f"shape audioRShaped {audioRShaped.shape}")
    print(" ")
    audioLInterp = np.zeros((audioLShaped.shape[0],audioLShaped.shape[1]*interpolation)).astype(np.float16)
    audioRInterp = np.zeros((audioRShaped.shape[0],audioRShaped.shape[1]*interpolation)).astype(np.float16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    if interpolation > 1:
        print(" ")
        callback_function(-1,-1, text_state = True, text_message = "Upsampling...")
        print(" ")
        audioLInterp = signal.resample(audioLShaped, audioLShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audioRInterp = signal.resample(audioRShaped, audioRShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        print(f"audioLInterp {audioLInterp.shape}")
        print(f"audioRInterp {audioRInterp.shape}")
        fs = fs*interpolation
        print(f"fs {fs}")
        size_frame = size_frame*interpolation
    else:
        audioLInterp = audioLShaped.astype(np.float16)
        audioRInterp = audioRShaped.astype(np.float16)
    callback_function(-1,-1, text_state = True, text_message = "Loading...")
    
    audioLInterp = audioLInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audioRInterp = audioRInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    
    audioLInterp = ((audioLInterp*32768 + 32768) * (res_height-1) / (65535)).astype(np.int16)
    audioRInterp = ((audioRInterp*32768 + 32768) * (res_width-1) / (65535)).astype(np.int16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    
    audioLInterp = np.clip(audioLInterp,0,res_height-1).astype(np.int16)
    audioRInterp = np.clip(audioRInterp,0,res_width-1).astype(np.int16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
      
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)

        for m in range(size_frame):
            frameData[audioLInterp[i,m],audioRInterp[i,m]] = True

        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
    
def generate_polar(output_name,input_audio,channel,fps, res_width, res_height,offset, note, interpolation,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
     
    if song.shape[0] == 2:
         
        if channel == "Both (Merge to mono)":
            audio = np.mean(song, axis = 0).T
            print(f"audiooooo {audio.shape}")
        elif channel == "Left":
            audio = song[0,:].T
        elif channel == "Right":
            audio = song[1,:].T
    else:
        audio = song.T
        print(f"audio {audio.shape}")
    
    print(f"shape audio {audio.shape}")
    
    # A4 ---> 2764.5
    polar_speed = note_to_polarSpeed(note)
    
    audio = (audio/np.max(np.max(abs(abs(audio))))).T ##TRANSPOSITION AND NORMALIZATION
    audio = np.clip(audio,-1,1)
    audio = audio + offset ###################################################ADDING OFFSET
    audio = (audio/np.max(np.max(abs(abs(audio))))) ##NORMALIZATION AGAIN
    print(" xd")
    print(f"audio {audio.shape}")

    erre = audio.astype(np.float16)
    theta = (np.linspace(0, polar_speed*len(erre)/fs, len(erre)) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision
    print(erre)
    print(theta)
    print(f"erre {erre.shape}")
    print(f"theta {theta.shape}")   
    
    audioL = (erre*np.sin(theta)).astype(np.float16)
    audioR = (erre*np.cos(theta)).astype(np.float16) ## 32768*0.95
     
    print(f"audioL {audioL.shape}")
    print(audioL)
    print(f"audioR {audioR.shape}")
    print(audioR)

     
    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(len(audioL)/size_frame))
     
    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    print(f"shape audioL {audioL.shape}")
    
    extra_margin = 50
    audioL = np.pad(audioL,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audioR = np.pad(audioR,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    print(f"shape audioL {audioL.shape}")
    print(" ")
    audioLShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audioRShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(" ")
    for i in range(n_frames):
        audioLShaped[i,:] = audioL[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audioRShaped[i,:] = audioR[i*size_frame : (i+1)*size_frame + extra_margin*2]
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(f"shape audioRShaped {audioRShaped.shape}")
    print(" ")
    audioLInterp = np.zeros((audioLShaped.shape[0],audioLShaped.shape[1]*interpolation)).astype(np.float16)
    audioRInterp = np.zeros((audioRShaped.shape[0],audioRShaped.shape[1]*interpolation)).astype(np.float16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    if interpolation > 1:
        print(" ")
        callback_function(-1,-1, text_state = True, text_message = "Upsampling...")
        print(" ")
        audioLInterp = signal.resample(audioLShaped, audioLShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audioRInterp = signal.resample(audioRShaped, audioRShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        print(f"audioLInterp {audioLInterp.shape}")
        print(f"audioRInterp {audioRInterp.shape}")
        fs = fs*interpolation
        print(f"fs {fs}")
        size_frame = size_frame*interpolation
    else:
        audioLInterp = audioLShaped.astype(np.float16)
        audioRInterp = audioRShaped.astype(np.float16)
    callback_function(-1,-1, text_state = True, text_message = "Loading...")
    
    audioLInterp = audioLInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audioRInterp = audioRInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    
    audioLInterp = ((audioLInterp*31130 + 32768) * (res_height-1) / (65535)).astype(np.int16) ## 31130 = 32768*0.95
    audioRInterp = ((audioRInterp*31130 + 32768) * (res_width-1) / (65535)).astype(np.int16)  ## 31130 = 32768*0.95
    
    audioLInterp = np.clip(audioLInterp,0,res_height-1).astype(np.int16)
    audioRInterp = np.clip(audioRInterp,0,res_width-1).astype(np.int16)
        
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)

        for m in range(size_frame):
            frameData[audioLInterp[i,m],audioRInterp[i,m]] = True

        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
    
def generate_polar_stereo(output_name,input_audio,fps, res_width, res_height,offset, note, interpolation,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    song = song/np.max(np.max(abs(abs(song)))) ##TRANSPOSITION AND NORMALIZATION
    song = np.clip(song,-1,1)
    song = song + offset ###################################################ADDING OFFSET
    song = (song/np.max(np.max(abs(abs(song))))) ##NORMALIZATION AGAIN
    
    audio0 = song[0,:].T
    audio1 = song[1,:].T
    
    # A4 ---> 2764.5
    polar_speed = note_to_polarSpeed(note)

    #erre = audioInterp.astype(np.float16)
    theta = (np.linspace(0, polar_speed*len(audio0)/fs, len(audio0)) % 2*np.pi).astype(np.float16) ## mod 2pi because numbers get big and with float16 they lack precision
    #print(erre)
    print(theta)
    #print(f"erre {erre.shape}")
    print(f"theta {theta.shape}")   
    
    audioL = (audio0*np.sin(theta)).astype(np.float16) ## 32768*0.95
    audioR = (audio1*np.cos(theta)).astype(np.float16) ## 32768*0.95
    print(" ")
    
    ########### ROTATION 45 DEG ######################
    sqrt2_over_2 = np.sqrt(2) / 2
    audioLr = (audioR*sqrt2_over_2 + audioL*sqrt2_over_2).astype(np.float16)
    audioRr = (-audioR*sqrt2_over_2 + audioL*sqrt2_over_2).astype(np.float16)
    print(" ")
    audioL = audioLr
    audioR = audioRr
    ##################################################
     
    print(f"audioL {audioL.shape}")
    print(audioL)
    print(f"audioR {audioR.shape}")
    print(audioR)

    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(len(audioL)/size_frame))
    
    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    print(f"shape audioL {audioL.shape}")
    
    extra_margin = 50
    audioL = np.pad(audioL,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audioR = np.pad(audioR,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    print(f"shape audioL {audioL.shape}")
    print(" ")
    audioLShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audioRShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(" ")
    for i in range(n_frames):
        audioLShaped[i,:] = audioL[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audioRShaped[i,:] = audioR[i*size_frame : (i+1)*size_frame + extra_margin*2]
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(f"shape audioRShaped {audioRShaped.shape}")
    print(" ")
    audioLInterp = np.zeros((audioLShaped.shape[0],audioLShaped.shape[1]*interpolation)).astype(np.float16)
    audioRInterp = np.zeros((audioRShaped.shape[0],audioRShaped.shape[1]*interpolation)).astype(np.float16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    if interpolation > 1:
        print(" ")
        callback_function(-1,-1, text_state = True, text_message = "Upsampling...")
        print(" ")
        audioLInterp = signal.resample(audioLShaped, audioLShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audioRInterp = signal.resample(audioRShaped, audioRShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        print(f"audioLInterp {audioLInterp.shape}")
        print(f"audioRInterp {audioRInterp.shape}")
        fs = fs*interpolation
        print(f"fs {fs}")
        size_frame = size_frame*interpolation
    else:
        audioLInterp = audioLShaped.astype(np.float16)
        audioRInterp = audioRShaped.astype(np.float16)
    callback_function(-1,-1, text_state = True, text_message = "Loading...")
    
    audioLInterp = audioLInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audioRInterp = audioRInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    
    audioLInterp = ((audioLInterp*31130 + 32768) * (res_height-1) / (65535)).astype(np.int16) ## 31130 = 32768*0.95
    audioRInterp = ((audioRInterp*31130 + 32768) * (res_width-1) / (65535)).astype(np.int16)  ## 31130 = 32768*0.95
    
    audioLInterp = np.clip(audioLInterp,0,res_height-1).astype(np.int16)
    audioRInterp = np.clip(audioRInterp,0,res_width-1).astype(np.int16)
        
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Generate and save each frame as an image
    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)

        for m in range(size_frame):
            frameData[audioLInterp[i,m],audioRInterp[i,m]] = True

        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_recurrence(output_name,input_audio,channel,fps, res_width, res_height, note, threshold, thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
     
    if song.shape[0] == 2:
         
        if channel == "Both (Stereo)":
            audioL = song[0,:].T
            audioR = song[1,:].T
            print(f"audioooooL {audioL.shape}")
            print(f"audioooooR {audioR.shape}")
        elif channel == "Both (Merge to mono)":
            audioL = np.mean(song, axis = 0).T
            audioR = np.mean(song, axis = 0).T
            print(f"audioooooL {audioL.shape}")
            print(f"audioooooR {audioR.shape}")
        elif channel == "Left":
            audioL = song[0,:].T
            audioR = song[0,:].T
        elif channel == "Right":
            audioL = song[1,:].T
            audioR = song[1,:].T
    else:
        audioL = song.T
        audioR = song.T
        print(f"audioL {audioL.shape}")
        print(f"audioR {audioR.shape}")

    gmax = np.max([np.max(np.abs(audioL)), np.max(np.abs(audioR))])
    #print(gmax)
    audioL = (audioL/gmax).T ##TRANSPOSITION AND NORMALIZATION
    audioR = (audioR/gmax).T ##TRANSPOSITION AND NORMALIZATION
    #print(audioL)
    
    duration = len(audioL)/fs
    #note = "C2"
    freq_tune = note_to_frequency(note)
    speed = fs/freq_tune  #FLOAT
    fps_falso = fs/speed  #FLOAT
    print("yes")
    size_frame = int(np.round(fs/fps))
    n_frames_falso = int(np.ceil(len(audioL)/speed))
    n_frames = round(duration*fps)
    print("yes")
    indexes = np.linspace(0, n_frames_falso - 1, n_frames) ## FPS ARE NOW 60
    indexes2 = [round(x) for x in indexes]
    print("yes")
    size_frameL = speed #float 
    size_frameR = speed #float 
    print("yes")
    if size_frameL < res_height: #JUST IN CASE THE HEIGHT OR WIDTH IS BIGGER THEN THE # OF SAMPLES IN THE SEGMENT
        print(f"size_frameL {size_frameL}")
        print(f"audioL {audioL.shape}")
        audioL = signal.resample(audioL, int(len(audioL)*res_height/size_frameL))
        print(f"audioL {audioL.shape}")
        audioL = np.clip(audioL,-1,1)
        size_frameL = res_height
        print("aja")
    print("yes")
    if size_frameR < res_width:
        print(f"size_frameR {size_frameR}")
        print(f"audioR {audioR.shape}")
        audioR = signal.resample(audioR, int(len(audioR)*res_width/size_frameR))
        print(f"audioR {audioR.shape}")
        audioR = np.clip(audioR,-1,1)
        size_frameR = res_width
        print("aja")
    print("yes")

    #print(f"audioL {audioL.shape}")
    #print(f"audioR {audioR.shape}")
    print(f"n_frames_falso {n_frames_falso}")
    print(f"n_frames {n_frames}")
    audioL = np.pad(audioL, (0, int(size_frameL*n_frames_falso) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frameR*n_frames_falso) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    print("yesuu")
    print(size_frameR)
    print(size_frameL)
    xaxis = np.linspace(0,res_width - 1,int(size_frameR)).astype(int)
    print("yes")
    print(xaxis)
    yaxis = np.linspace(0,res_height - 1,int(size_frameL)).astype(int)
    print(yaxis)
    print("yes")
    print(indexes2)
    print(f"xaxis {xaxis.shape}")
    print(f"yaxis {yaxis.shape}")
    #print(xaxis)
    #print(yaxis)
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frameData = np.zeros((res_width, res_height), dtype=bool)
    j = 0
    print("no")
    for i in indexes2:
        #print("yes")
        audioLseg = audioL[round(i*size_frameL) : round(i*size_frameL)+int(size_frameL)]
        audioRseg = audioR[round(i*size_frameR) : round(i*size_frameR)+int(size_frameR)]
        #print(audioLseg)
        #print("yes")
        #print(f"audioLseg {audioLseg.shape}")
        #print(f"audioRseg {audioRseg.shape}")

        audioLseg = audioLseg[:, np.newaxis]
        audioRseg = audioRseg[:, np.newaxis]
        #print(audioLseg)
        #print(f"audioLseg {audioLseg.shape}")
        distances = np.abs(audioLseg.T - audioRseg)
        #print(f"distances {distances.shape}")
        #print("yes")
        #print(f"frameData {frameData.shape}")
        if threshold >= 0:
            frameData[xaxis[:, np.newaxis], yaxis] = (distances < threshold)
        else:
            frameData[xaxis[:, np.newaxis], yaxis] = ~(distances < -threshold)
        #print("yes")
        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        j += 1
        print(f"{j+1}/{n_frames}")
        callback_function(j+1,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def generate_chladni(output_name,input_audio,channel,fps, res_width, res_height, mode, zoom, exp_filter, threshold, thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)

    if song.shape[0] == 2:
        if channel == "Both (Stereo)":
            audioL = song[0,:].T
            audioR = song[1,:].T
            print(f"audioooooL {audioL.shape}")
            print(f"audioooooR {audioR.shape}")
        elif channel == "Both (Merge to mono)":
            audioL = np.mean(song, axis = 0).T
            audioR = np.mean(song, axis = 0).T
            print(f"audioooooL {audioL.shape}")
            print(f"audioooooR {audioR.shape}")
        elif channel == "Left":
            audioL = song[0,:].T
            audioR = song[0,:].T
        elif channel == "Right":
            audioL = song[1,:].T
            audioR = song[1,:].T
    else:
        audioL = song.T
        audioR = song.T
        print(f"audioL {audioL.shape}")
        print(f"audioR {audioR.shape}")

    #if fil: #HIGH-PASS FILTER FOR HAVING MORE TRANSIENTS
    N = 5
    h = np.cos(np.linspace(0,2*np.pi,N)) - 1
    h[int((N-1)/2)] = -sum(h) + h[int((N-1)/2)]
    audioL = np.convolve(audioL, h, mode = 'same')
    audioR = np.convolve(audioR, h, mode = 'same')

    gmax = np.max([np.max(np.abs(audioL)), np.max(np.abs(audioR))])
    #print(gmax)
    audioL = (audioL/gmax).T ##TRANSPOSITION AND NORMALIZATION
    audioR = (audioR/gmax).T ##TRANSPOSITION AND NORMALIZATION

    duration = len(audioL)/fs
    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(duration*fps))

    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    xaxis = np.linspace(1/res_width , res_width/zoom , int(res_width/2)).astype(np.float32) #solo el primer cuadrante
    yaxis = np.linspace(1/res_height , res_height/zoom , int(res_height/2)).astype(np.float32) #solo el primer cuadrante
    Xmap, Ymap = np.meshgrid(xaxis, yaxis)
    Xmap = np.pi * Xmap.astype(np.float32)
    Ymap = np.pi * Ymap.astype(np.float32)
    print(Xmap)
    print(Ymap)

    print(f"Xmap {Xmap.shape}")
    #print(xaxis)

    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]

    chladniQ1 = np.zeros((int(res_width/2), int(res_height/2)), dtype=bool)
    chladniQ2 = chladniQ1
    chladniQ3 = chladniQ1
    chladniQ4 = chladniQ1

    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frameData = np.zeros((res_width, res_height), dtype=bool)
    j = 0

    exp_filter = np.sqrt(exp_filter)

    envL_past = 0
    envR_past = 0
    for i in range(n_frames):
        envL_now = 10*np.max(audioL[round(i*size_frame) : round(i*size_frame)+int(size_frame)]) #DETERMINAR LA ENVOLVENTE
        envR_now = 10*np.max(audioR[round(i*size_frame) : round(i*size_frame)+int(size_frame)]) #DETERMINAR LA ENVOLVENTE

        envL = envL_now*(1-exp_filter) + envL_past*exp_filter
        envR = envR_now*(1-exp_filter) + envR_past*exp_filter

        envL_past = envL
        envR_past = envR

        a = 5*(2+np.sin(1+i/n_frames*duration))
        b = 5*(2-np.cos(1+i/n_frames*duration))

        match mode:
            case "Sine":
                trgXL = np.sin(Xmap / envL*a)
                trgXR = np.sin(Xmap / envR*b)
                trgYL = np.sin(Ymap / envL*a)
                trgYR = np.sin(Ymap / envR*b)
            case "Cosine":
                trgXL = np.cos(Xmap / envL*a)
                trgXR = np.cos(Xmap / envR*b)
                trgYL = np.cos(Ymap / envL*a)
                trgYR = np.cos(Ymap / envR*b)
            case "Tangent":
                trgXL = np.tan(Xmap / envL*a)
                trgXR = np.tan(Xmap / envR*b)
                trgYL = np.tan(Ymap / envL*a)
                trgYR = np.tan(Ymap / envR*b)
            case "Cotangent":
                trgXL = 1/np.tan(Xmap / envL*a)
                trgXR = 1/np.tan(Xmap / envR*b)
                trgYL = 1/np.tan(Ymap / envL*a)
                trgYR = 1/np.tan(Ymap / envR*b)
            case "Secant":
                trgXL = 1/np.cos(Xmap / envL*a)
                trgXR = 1/np.cos(Xmap / envR*b)
                trgYL = 1/np.cos(Ymap / envL*a)
                trgYR = 1/np.cos(Ymap / envR*b)
            case "Cosecant":
                trgXL = 1/np.sin(Xmap / envL*a)
                trgXR = 1/np.sin(Xmap / envR*b)
                trgYL = 1/np.sin(Ymap / envL*a)
                trgYR = 1/np.sin(Ymap / envR*b)

        if threshold >= 0:
            chladniQ1 = (np.abs(a * trgXL * trgYR + b * trgXR * trgYL) < threshold)
        else:
            chladniQ1 = ~(np.abs(a * trgXL * trgYR + b * trgXR * trgYL) < -threshold)

        chladniQ2 = np.fliplr(chladniQ1)
        chladniQ3 = np.flipud(chladniQ1)
        chladniQ4 = np.flipud(chladniQ2)

        frameData = np.vstack((np.hstack((chladniQ4, chladniQ3)), np.hstack((chladniQ2, chladniQ1))))

        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2

        frameData = frameData.astype(np.uint8) * 255

        ffmpeg_process.stdin.write(frameData)
        j += 1
        print(f"{j+1}/{n_frames}")
        callback_function(j+1,n_frames, text_state = False, text_message = " ")

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)

    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
     
def generate_poincare(output_name,input_audio,channel ,fps, res_width, res_height, delay, interpolation,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    song = song/np.max(np.max(abs(abs(song)))) ##TRANSPOSITION AND NORMALIZATION
    song = np.clip(song,-1,1)
    
    if song.shape[0] == 2:
         
        if channel == "Both (Merge to mono)":
            audio = np.mean(song, axis = 0).T
            print(f"audiooooo {audio.shape}")
        elif channel == "Left":
            audio = song[0,:].T
        elif channel == "Right":
            audio = song[1,:].T
    else:
        audio = song.T
        print(f"audio {audio.shape}")
    
    print(f"shape audio {audio.shape}")
    
    audioL = np.concatenate((audio[delay:].astype(np.float16), audio[:delay].astype(np.float16))).astype(np.float16)
    audioR = -audio.astype(np.float16)
    
    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(len(audioL)/size_frame))
     
    audioL = np.pad(audioL, (0, int(size_frame*n_frames) - len(audioL))) ## TO COMPLETE THE LAST FRAME
    audioR = np.pad(audioR, (0, int(size_frame*n_frames) - len(audioR))) ## TO COMPLETE THE LAST FRAME
    print(f"shape audioL {audioL.shape}")
    
    extra_margin = 50
    audioL = np.pad(audioL,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audioR = np.pad(audioR,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    print(f"shape audioL {audioL.shape}")
    print(" ")
    audioLShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audioRShaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(" ")
    for i in range(n_frames):
        audioLShaped[i,:] = audioL[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audioRShaped[i,:] = audioR[i*size_frame : (i+1)*size_frame + extra_margin*2]
    print(f"shape audioLShaped {audioLShaped.shape}")
    print(f"shape audioRShaped {audioRShaped.shape}")
    print(" ")
    audioLInterp = np.zeros((audioLShaped.shape[0],audioLShaped.shape[1]*interpolation)).astype(np.float16)
    audioRInterp = np.zeros((audioRShaped.shape[0],audioRShaped.shape[1]*interpolation)).astype(np.float16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    if interpolation > 1:
        print(" ")
        callback_function(-1,-1, text_state = True, text_message = "Upsampling...")
        print(" ")
        audioLInterp = signal.resample(audioLShaped, audioLShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audioRInterp = signal.resample(audioRShaped, audioRShaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        print(f"audioLInterp {audioLInterp.shape}")
        print(f"audioRInterp {audioRInterp.shape}")
        fs = fs*interpolation
        print(f"fs {fs}")
        size_frame = size_frame*interpolation
    else:
        audioLInterp = audioLShaped.astype(np.float16)
        audioRInterp = audioRShaped.astype(np.float16)
    callback_function(-1,-1, text_state = True, text_message = "Loading...")
    
    audioLInterp = audioLInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audioRInterp = audioRInterp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    
    audioLInterp = ((audioLInterp*32768 + 32768) * (res_height-1) / (65535)).astype(np.int16)
    audioRInterp = ((audioRInterp*32768 + 32768) * (res_width-1) / (65535)).astype(np.int16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
    
    audioLInterp = np.clip(audioLInterp,0,res_height-1).astype(np.int16)
    audioRInterp = np.clip(audioRInterp,0,res_width-1).astype(np.int16)
    print(f"audioLInterp {audioLInterp.shape}")
    print(f"audioRInterp {audioRInterp.shape}")
      
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)

        for m in range(size_frame):
            frameData[audioLInterp[i,m],audioRInterp[i,m]] = True

        #thickness = 1 ## REPEATS THE IMAGE SO IT'S THICKER
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0
    
def generate_delay_embed(output_name,input_audio,channel ,fps, res_width, res_height, delay1,delay2, beta_p, beta_s, alfa_p, alfa_s,interpolation,thickness,compression, callback_function):

    root, vidfor = os.path.splitext(output_name)

    song, fs = read_audio_samples(input_audio)
    song = song.T.astype(np.float16)
    song = song/np.max(np.max(abs(abs(song)))) ##TRANSPOSITION AND NORMALIZATION
    song = np.clip(song,-1,1)
    
    if song.shape[0] == 2:
        if channel == "Both (Stereo)":
            audio0 = -np.mean(song, axis = 0).T.astype(np.float16)
            audio1 = np.concatenate((song[0,delay1:].astype(np.float16), song[0,:delay1].astype(np.float16))).T.astype(np.float16)
            audio2 = np.concatenate((song[1,delay2:].astype(np.float16), song[1,:delay2].astype(np.float16))).T.astype(np.float16)
        elif channel == "Both (Merge to mono)":
            audio0 = -np.mean(song, axis = 0).T.astype(np.float16)
            audio1 = np.concatenate((audio0[delay1:].astype(np.float16), audio0[:delay1].astype(np.float16))).astype(np.float16)
            audio2 = np.concatenate((audio0[delay2:].astype(np.float16), audio0[:delay2].astype(np.float16))).astype(np.float16)
        elif channel == "Left":
            audio0 = -song[0,:].T.astype(np.float16)
            audio1 = np.concatenate((audio0[delay1:].astype(np.float16), audio0[:delay1].astype(np.float16))).astype(np.float16)
            audio2 = np.concatenate((audio0[delay2:].astype(np.float16), audio0[:delay2].astype(np.float16))).astype(np.float16)
        elif channel == "Right":
            audio0 = -song[1,:].T.astype(np.float16)
            audio1 = np.concatenate((audio0[delay1:].astype(np.float16), audio0[:delay1].astype(np.float16))).astype(np.float16)
            audio2 = np.concatenate((audio0[delay2:].astype(np.float16), audio0[:delay2].astype(np.float16))).astype(np.float16)
    else:
        audio0 = -song.T.astype(np.float16)
        audio1 = np.concatenate((audio0[delay1:].astype(np.float16), audio0[:delay1].astype(np.float16))).astype(np.float16)
        audio2 = np.concatenate((audio0[delay2:].astype(np.float16), audio0[:delay2].astype(np.float16))).astype(np.float16)
  
    size_frame = int(np.round(fs/fps))
    n_frames = int(np.ceil(len(audio1)/size_frame))
    
    audio2 = np.pad(audio2, (0, int(size_frame*n_frames) - len(audio2))) ## TO COMPLETE THE LAST FRAME
    audio1 = np.pad(audio1, (0, int(size_frame*n_frames) - len(audio1))) ## TO COMPLETE THE LAST FRAME
    audio0 = np.pad(audio0, (0, int(size_frame*n_frames) - len(audio0))) ## TO COMPLETE THE LAST FRAME
    print(f"shape audio1 {audio1.shape}")
    
    extra_margin = 50
    audio2 = np.pad(audio2,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audio1 = np.pad(audio1,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    audio0 = np.pad(audio0,(extra_margin,extra_margin)) ## TO ADD 100 SAMPLES AT THE START TO LATER REMOVE FOR RESAMPLING
    print(f"shape audio1 {audio1.shape}")
    print(" ")
    audio2Shaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audio1Shaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    audio0Shaped = np.zeros((n_frames,size_frame + extra_margin*2)).astype(np.float16) #chopping + 200 for margin
    print(f"shape audio1Shaped {audio1Shaped.shape}")
    print(" ")
    for i in range(n_frames):
        audio2Shaped[i,:] = audio2[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audio1Shaped[i,:] = audio1[i*size_frame : (i+1)*size_frame + extra_margin*2]
        audio0Shaped[i,:] = audio0[i*size_frame : (i+1)*size_frame + extra_margin*2]
    print(f"shape audio1Shaped {audio1Shaped.shape}")
    print(f"shape audio0Shaped {audio0Shaped.shape}")
    print(" ")
    audio2Interp = np.zeros((audio2Shaped.shape[0],audio2Shaped.shape[1]*interpolation)).astype(np.float16)
    audio1Interp = np.zeros((audio1Shaped.shape[0],audio1Shaped.shape[1]*interpolation)).astype(np.float16)
    audio0Interp = np.zeros((audio0Shaped.shape[0],audio0Shaped.shape[1]*interpolation)).astype(np.float16)
    print(f"audio1Interp {audio1Interp.shape}")
    print(f"audio0Interp {audio0Interp.shape}")
    if interpolation > 1:
        print(" ")
        callback_function(-1,-1, text_state = True, text_message = "Upsampling...")
        print(" ")
        audio2Interp = signal.resample(audio2Shaped, audio2Shaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audio1Interp = signal.resample(audio1Shaped, audio1Shaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        audio0Interp = signal.resample(audio0Shaped, audio0Shaped.shape[1]*interpolation, axis = 1).astype(np.float16)
        print(f"audio1Interp {audio1Interp.shape}")
        print(f"audio0Interp {audio0Interp.shape}")
        fs = fs*interpolation
        print(f"fs {fs}")
        size_frame = size_frame*interpolation
    else:
        audio2Interp = audio2Shaped.astype(np.float16)
        audio1Interp = audio1Shaped.astype(np.float16)
        audio0Interp = audio0Shaped.astype(np.float16)
    callback_function(-1,-1, text_state = True, text_message = "Loading...")
    
    audio2Interp = audio2Interp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audio1Interp = audio1Interp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    audio0Interp = audio0Interp[:, extra_margin*interpolation:-extra_margin*interpolation] ## TO REMOVE THE 100*interpolation SAMPLES FOR RESAMPLING
    print(f"audio1Interp {audio1Interp.shape}")
    print(f"audio0Interp {audio0Interp.shape}")
    
    #audio1Interp = ((audio1Interp*32768 + 32768) * (res_height-1) / (65535)).astype(np.int16)
    #audio0Interp = ((audio0Interp*32768 + 32768) * (res_width-1) / (65535)).astype(np.int16)
    #print(f"audio1Interp {audio1Interp.shape}")
    #print(f"audio0Interp {audio0Interp.shape}")
    
    #audio1Interp = np.clip(audio1Interp,0,res_height-1).astype(np.int16)
    #audio0Interp = np.clip(audio0Interp,0,res_width-1).astype(np.int16)
    #print(f"audio1Interp {audio1Interp.shape}")
    #print(f"audio0Interp {audio0Interp.shape}")
      
    cmd = [
        FFMPEG,
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-s', '{}x{}'.format(res_width, res_height),
        '-pix_fmt', 'gray',  # Use grayscale pixel format
        '-r', str(fps),  # Frames per second
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',  # Video codec
        '-preset', 'medium',  # Encoding speed vs compression ratio
        '-crf', str(compression),  # Constant Rate Factor (0-51): Lower values mean better quality
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        'resources/temporary_file.mp4'
    ]
    
    ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    beta = beta_p/180*np.pi
    alfa = alfa_p/180*np.pi
    # Generate and save each frame as an image
    for i in range(n_frames):
        frameData = np.zeros((res_height, res_width), dtype=bool)
        beta += 2*np.pi/fps*beta_s
        alfa += 2*np.pi/fps*alfa_s
        x_coo, y_coo = rotate_and_project(audio0Interp[i],audio1Interp[i],audio2Interp[i],alfa,beta,res_width,res_height)
        for m in range(size_frame):
            frameData[int(y_coo[m]), int(x_coo[m])] = True
        if thickness > 1:
            for th in range(thickness - 1):
                shifted = np.roll(frameData, shift=-1, axis=0) ##SHIFTS THE MATRIX UPWARDS
                shifted[-1, :] = False ## CLEARS BOTTOM ROW
                #frameData = (frameData + shifted)
                shifted2 = np.roll(frameData, shift=-1, axis=1) ##SHIFTS THE MATRIX TO THE RIGHT
                shifted2[:, -1] = False ## CLEARS LAST COLUMN
                frameData = frameData | shifted | shifted2 
        
        frameData = frameData.astype(np.uint8) * 255
        
        ffmpeg_process.stdin.write(frameData)
        print(f"{i+1}/{n_frames}")
        callback_function(i,n_frames, text_state = False, text_message = " ")
        
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    callback_function(i,n_frames, text_state = True, text_message = "Joining frames...")
    convert_vid(input_audio, output_name, vidfor)
        
    print(f"Video saved to {output_name}")
    os.remove("resources/temporary_file.mp4")
    callback_function(i,n_frames, text_state = True, text_message = "Done, my dood!")
    return 0

def rotate_and_project(a0, a1, a2, alpha, beta, W, H):
    P = np.vstack((a0, a1, a2)) 

    Ry = np.array([
        [np.cos(alpha), 0, np.sin(alpha)],
        [0, 1, 0],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(beta), -np.sin(beta)],
        [0, np.sin(beta), np.cos(beta)]
    ])

    R = Rx @ Ry
    Protated = R @ P 

    x2D, y2D = Protated[0], Protated[1]

    x_coo = (x2D*W/3 + W/2).astype(int)
    y_coo = (y2D*H/3 + H/2).astype(int)

    x_coo = np.clip(x_coo, 0, W - 1).astype(int)
    y_coo = np.clip(y_coo, 0, H - 1).astype(int)

    return x_coo, y_coo

def note_to_frequency(note):
    if note.lstrip('-').replace('.', '', 1).isdigit():    
        frequency = float(note)
    else:
        note_to_midi = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                        'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6,
                        'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
                        'B': 11, 'Cb': 11, 'B#': 0,
                        'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
                        'e': 4, 'fb': 4, 'e#': 5, 'f': 5, 'f#': 6, 'gb': 6,
                        'g': 7, 'g#': 8, 'ab': 8, 'a': 9, 'a#': 10, 'bb': 10,
                        'b': 11, 'cb': 11, 'b#': 0}

        note_name, octave_str = note[:-1], note[-1]
        midi_note = note_to_midi[note_name] + (int(octave_str) + 1) * 12
        frequency = 440 * (2 ** ((midi_note - 69) / 12))

    return frequency
    
def note_to_polarSpeed(note):
    #A4 ---> 2764.55 (?????)
    frequency = note_to_frequency(note)*2

    return frequency


#################################################
################# GUI FUNCTIONS #################
#################################################

def create_input_widgets_num(master, label_text, variable, row, tip):
    tk.Label(master, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(master, textvariable=variable, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    entry.grid(row=row, column=1, padx=10, pady=5, sticky="we")
    entry.config(width=10)

    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def create_input_widgets(master, label_text, variable, row, tip):
    tk.Label(master, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(master, textvariable=variable, validate="key")
    entry.grid(row=row, column=1, padx=10, pady=5, sticky="we")
    entry.config(width=10)

    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def create_readonly_dropdown(master, variable, row, values):
    combobox_var = tk.StringVar()
    combobox = ttk.Combobox(master, textvariable=variable, values=values, state='readonly')
    combobox.grid(row=row, column=2, padx=0, pady=5, sticky="w")
    combobox.current(0)
    combobox.config(width=6)

def create_checkbutton(master, label_text, variable, row, tip):
    checkbutton = tk.Checkbutton(master, text=label_text, variable=variable)
    checkbutton.grid(row=row, column=1, padx=10, pady=5, sticky="w")


    tip_label = tk.Label(master, text=tip, font=("Helvetica", 10), fg="gray", anchor="w", justify="left")
    tip_label.grid(row=row, column=2, padx=5, pady=5, sticky="w")

def create_combobox(master, label_text, variable, row, values, tip=None, readonly=False):
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

def create_combobox_dual(master, label_text, variable, divider, variable2, row, values, values2, tip=None):
    label = tk.Label(master, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky='e')

    combobox = ttk.Combobox(master, textvariable=variable, values=values, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    combobox.grid(row=row, column=1, padx=10, pady=5, sticky='w')
    combobox.config(width=6)

    label2 = tk.Label(master, text=divider)
    label2.grid(row=row, column=1, padx=100, pady=5, sticky='w')

    combobox2 = ttk.Combobox(master, textvariable=variable2, values=values2, validate="key", validatecommand=(master.register(validate_numeric), "%P"))
    combobox2.grid(row=row, column=1, padx=10, pady=5, sticky='e')
    combobox2.config(width=6)
    if tip:
        tooltip = tk.Label(master, text=tip, font=("Helvetica", 10), fg='gray', anchor="w", justify="left")
        tooltip.grid(row=row, column=2, padx=5, pady=5, sticky='w')

def validate_numeric(value):
    if value == '' or value == '-':
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False

def create_file_input_row(parent, label_text, row, path_var=None):
    label = tk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

    entry = tk.Entry(parent, textvariable=path_var)
    entry.grid(row=row, column=1, columnspan = 2, padx=92, pady=5, sticky="w")
    entry.config(width=60)

    def browse():
        file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.aac *.flac *.ogg *.opus *.wma *.m4a"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            if path_var is not None:
                path_var.set(file_path)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
            entry.xview_moveto(1)  #SHOW THE NAME OF THE FILE IN THE PATH

    button = tk.Button(parent, text="Browse", command=browse)
    button.grid(row=row, column=1, padx=5, pady=5, sticky="w")

    return entry

def create_file_output_row(parent, label_text, row, path_var=None):
    label = tk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

    entry = tk.Entry(parent, textvariable=path_var)
    entry.grid(row=row, column=1, columnspan = 2, padx=92, pady=5, sticky="w")
    entry.config(width=60)

    def browse():
        file_path = filedialog.asksaveasfilename(
            title="Save video as",
            defaultextension=".mp4",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.webm *.webp *.gif *.flv *.mkv *.mov *.wmv *.3gp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            if path_var is not None:
                path_var.set(file_path)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
            entry.xview_moveto(1)  #SHOW THE NAME OF THE FILE IN THE PATH

    button = tk.Button(parent, text="Browse", command=browse)
    button.grid(row=row, column=1, padx=5, pady=5, sticky="w")

    return entry


#################################################
#################### WINDOWS ####################
#################################################

class SpectrumWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    xlow_values = [1,20,300]
    xhigh_values = [600,1000,2000,5000,13000,20000]
    style_values = ["Just Points", "Curve", "Filled Spectrum"]

    def __init__(self, master):
        self.master = master
        self.master.title("Linear Spectrum Visualizer v0.19 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=1920)
        self.res_height = tk.IntVar(value=540)
        self.t_smoothing = tk.IntVar(value=1)
        self.xlow = tk.IntVar(value=1)
        self.xhigh = tk.IntVar(value=13000)
        self.limt_junk = tk.BooleanVar(value=True)
        self.attenuation_steep = tk.DoubleVar(value=0.5)
        self.junk_threshold = tk.DoubleVar(value=2)
        self.threshold_steep = tk.DoubleVar(value=10)
        self.style = tk.StringVar(value="Filled Spectrum")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip="", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Time Smoothing:", self.t_smoothing, row=row_num, tip="Gives you a curve that doesn't move that violently. Whole number.")
        row_num += 1
        create_combobox_dual(self.master, "Frequency Limits:", self.xlow, "-", self.xhigh, row=row_num, values=self.xlow_values, values2=self.xhigh_values, tip="Lower and higher frequency limits in Hz, respectively.\nFrom 1Hz to half the sample rate of the audio.")
        row_num += 1
        create_input_widgets_num(self.master, "Mid/High Boost:", self.attenuation_steep, row=row_num, tip="This will boost everything but the low end. You can enter a negative\nvalue for some crazy results, but going below -10 has no purpose.")
        row_num += 1
        create_checkbutton(self.master, "Expander", self.limt_junk, row=row_num, tip="This will reduce the intensity of small amplitudes.")
        row_num += 1
        create_input_widgets_num(self.master, "Expand Threshold:", self.junk_threshold, row=row_num, tip="The bigger this value, the bigger amplitudes have to be to not\nbe reduced. Doesn't have to be a whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Expand Steepness:", self.threshold_steep, row=row_num, tip="This will make the transition between amplitudes being reduced\nor boosted more abrupt. Doesn't have to be a whole number.")
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up. Whole number.\nWill make the render slower the higher you go")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1

        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1

        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(text=f"Loading...")
            self.loading_label.config(fg="blue")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            t_smoothing = self.t_smoothing.get()
            xlow = self.xlow.get()
            xhigh = self.xhigh.get()
            limt_junk = self.limt_junk.get()
            attenuation_steep = self.attenuation_steep.get()
            junk_threshold = self.junk_threshold.get()
            threshold_steep = self.threshold_steep.get()
            style = self.style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if t_smoothing <= 0 or (t_smoothing%1) != 0:
                self.loading_label.config(text=f"Error! Time smoothing must be a positive whole number.")
                error_flag = True
            if xlow <= 0 or (xlow % 1) != 0 or xhigh <= 0 or (xhigh % 1) != 0 or (xlow >= xhigh):
                self.loading_label.config(text=f"Error! Frequency values must be positive whole numbers.\nThe left one has to be lower than the right one.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_spectrum(output_name,input_audio,channel,fps,res_width,res_height,t_smoothing,xlow,xhigh,limt_junk,attenuation_steep,junk_threshold,threshold_steep,style,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class SpectrumdBWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    xlow_values = [1,20,300]
    xhigh_values = [600,1000,2000,5000,13000,20000]
    style_values = ["Just Points", "Curve", "Filled Spectrum"]

    def __init__(self, master):
        self.master = master
        self.master.title("Linear Spectrum Visualizer (dB) v0.12 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=1920)
        self.res_height = tk.IntVar(value=540)
        self.t_smoothing = tk.IntVar(value=1)
        self.xlow = tk.IntVar(value=1)
        self.xhigh = tk.IntVar(value=13000)
        self.min_dB = tk.DoubleVar(value=-80)
        self.style = tk.StringVar(value="Filled Spectrum")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip="", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Time Smoothing:", self.t_smoothing, row=row_num, tip="Gives you a curve that doesn't move that violently. Whole number.")
        row_num += 1
        create_combobox_dual(self.master, "Frequency Limits:", self.xlow, "-", self.xhigh, row=row_num, values=self.xlow_values, values2=self.xhigh_values, tip="Lower and higher frequency limits in Hz, respectively.\nFrom 1Hz to half the sample rate of the audio.")
        row_num += 1
        create_input_widgets_num(self.master, "Spectrum Floor:", self.min_dB, row=row_num, tip="Minimum value to display in dB. Less than 0dB")
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up. Whole number.\nWill make the render slower the higher you go")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1


        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)


        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(text=f"Loading...")
            self.loading_label.config(fg="blue")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            t_smoothing = self.t_smoothing.get()
            xlow = self.xlow.get()
            xhigh = self.xhigh.get()
            min_dB = self.min_dB.get()
            style = self.style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if t_smoothing <= 0 or (t_smoothing % 1) != 0:
                self.loading_label.config(text=f"Error! Time smoothing must be a positive whole number.")
                error_flag = True
            if xlow <= 0 or (xlow % 1) != 0 or xhigh <= 0 or (xhigh % 1) != 0 or (xlow >= xhigh):
                self.loading_label.config(text=f"Error! Frequency values must be positive whole numbers.\nThe left one has to be lower than the right one.")
                error_flag = True
            if min_dB >= 0:
                self.loading_label.config(text=f"Error! Spectrum floor has to be lower than 0dB.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_spectrum_dB(output_name,input_audio, channel, fps, res_width, res_height, t_smoothing, xlow, xhigh, min_dB, style, thickness, compression, self.update_loading_label)

        except Exception:
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class SpecBalanceWindow:
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    xlow_values = [1,20,300]
    xhigh_values = [600,1000,2000,5000,13000,20000]
    style_values = ["Just Points", "Curve", "Filled Spectrum"]

    def __init__(self, master):
        self.master = master
        self.master.title("Linear Spectral Balance Visualizer v0.03 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.t_smoothing = tk.IntVar(value=1)
        self.xlow = tk.IntVar(value=1)
        self.xhigh = tk.IntVar(value=13000)
        self.style = tk.StringVar(value="Curve")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Time Smoothing:", self.t_smoothing, row=row_num, tip="Gives you a curve that doesn't move that violently. Whole number.")
        row_num += 1
        create_combobox_dual(self.master, "Frequency Limits:", self.xlow, "-", self.xhigh, row=row_num, values=self.xlow_values, values2=self.xhigh_values, tip="Lower and higher frequency limits in Hz, respectively.\nFrom 1Hz to half the sample rate of the audio.")
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up. Whole number.\nWill make the render slower the higher you go")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1


        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1


        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(text=f"Loading...")
            self.loading_label.config(fg="blue")
            self.master.update()

            # Get values from Entry widgets and perform the final action
            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            t_smoothing = self.t_smoothing.get()
            xlow = self.xlow.get()
            xhigh = self.xhigh.get()
            style = self.style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if t_smoothing <= 0 or (t_smoothing%1) != 0:
                self.loading_label.config(text=f"Error! Time smoothing must be a positive whole number.")
                error_flag = True
            if xlow <= 0 or (xlow % 1) != 0 or xhigh <= 0 or (xhigh % 1) != 0 or (xlow >= xhigh):
                self.loading_label.config(text=f"Error! Frequency values must be positive whole numbers.\nThe left one has to be lower than the right one.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_spec_balance(output_name,input_audio,fps,res_width,res_height,t_smoothing,xlow,xhigh,style,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class HistogramWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    style_values = ["Just Points", "Curve", "Filled Histogram"]
    curve_style_values = ["Flat", "Linear", "FFT Resample"]

    def __init__(self, master):
        self.master = master
        self.master.title("Histogram Visualizer v0.04 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.size_frame = tk.IntVar(value=1000)
        self.bars = tk.IntVar(value=101)
        self.sensitivity = tk.DoubleVar(value=0.1)
        self.curve_style = tk.StringVar(value="Flat")
        self.style = tk.StringVar(value="Filled Histogram")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip="", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Window Size:", self.size_frame, row=row_num, tip="Number of samples of the histogram. Whole number.\nIf it's too low, it will be calculated with the FPS and sample rate.")
        row_num += 1
        create_input_widgets_num(self.master, "Bars:", self.bars, row=row_num, tip="Number of bars of the histogram. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Sensitivity:", self.sensitivity, row=row_num, tip="Makes smaller values more visible. Non-negative value.")
        row_num += 1
        create_combobox(self.master, "Curve Style:", self.curve_style, row=row_num, values=self.curve_style_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up. Whole number.\nWill make the render slower the higher you go")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1

        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1

        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(text=f"Loading...")
            self.loading_label.config(fg="blue")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            size_frame = self.size_frame.get()
            bars = self.bars.get()
            sensitivity = self.sensitivity.get()
            style = self.style.get()
            curve_style = self.curve_style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if size_frame <= 0 or (size_frame%1) != 0:
                self.loading_label.config(text=f"Error! Window size must be a positive whole number.")
                error_flag = True
            if bars <= 0 or (bars % 1) != 0:
                self.loading_label.config(text=f"Error! Number of bars must be a positive whole number.")
                error_flag = True
            if sensitivity < 0:
                self.loading_label.config(text=f"Error! Sensitivity must be a non-negative number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_histogram(output_name,input_audio,channel,fps,res_width,res_height, size_frame, bars, sensitivity, curve_style, style,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class WaveformWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    style_values = ["Just Points", "Curve", "Filled Waveform"]

    def __init__(self, master):
        self.master = master
        self.master.title("Short Waveform Visualizer v0.15 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps_2 = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=1920)
        self.res_height = tk.IntVar(value=540)
        self.note = tk.StringVar(value="C2")
        self.window_size = tk.IntVar(value=3000)
        self.style = tk.StringVar(value="Curve")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps_2, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets(self.master, "Tuning:", self.note, row=row_num, tip="Set a note to tune the oscilloscope to.\nYou can enter the name of a note or its fundamental frequency in Hz.")
        row_num += 1
        create_input_widgets_num(self.master, "Window Size:", self.window_size, row=row_num, tip="The smaller this is, the faster the waveform will move. Whole number.\nRecommended minimum is the width of the video.\nFor higher than ~20000 I recommend using the long waveform.")
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1

        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)

        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def validate_numeric(self, value):
        try:
            if not value:
                return True
            float(value)
            return True
        except ValueError:
            return False

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps_2 = self.fps_2.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            note = self.note.get()
            window_size = self.window_size.get()
            style = self.style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps_2 <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True

            if note.lstrip('-').replace('.', '', 1).isdigit():
                if float(note) <= 0:
                    self.loading_label.config(text=f"Error! Tuning frequency must be a positive number.")
                    error_flag = True
            else:
                if len(note) == 2:
                    if not note[0].lower() in 'abcdefg' or not note[1].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                elif len(note) == 3:
                    if not note[0].lower() in 'abcdefg' or not (note[1] == "#" or note[1] == "b") or not note[2].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                else:
                    self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                    error_flag = True

            if window_size <= 0 or (window_size % 1) != 0:
                self.loading_label.config(text=f"Error! Window size must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_waveform(output_name,input_audio, channel, fps_2, res_width, res_height, note, window_size, style, thickness, compression, self.update_loading_label)

        except Exception:
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class LongWaveformWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [480,720,960,1024,1280,1366,1440,1080,1920,2560,3840]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    style_values = ["Just Points", "Curve", "Filled Waveform"]

    def __init__(self, master):
        self.master = master
        self.master.title("Long Waveform Visualizer v0.05 by Aaron F. Bianchi")


        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=1920)
        self.res_height = tk.IntVar(value=540)
        self.window_size = tk.IntVar(value=400000)
        self.style = tk.StringVar(value="Curve")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width, "x", self.res_height, row=row_num, values=self.width_values, values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Window Size:", self.window_size, row=row_num, tip="The smaller this is, the faster the waveform will move. Whole number.")
        row_num += 1
        create_combobox(self.master, "Drawing Style:", self.style, row=row_num, values=self.style_values, tip=" ", readonly=True)
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1


        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)


        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def validate_numeric(self, value):
        try:
            if not value:
                return True
            float(value)
            return True
        except ValueError:
            return False

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            window_size = self.window_size.get()
            style = self.style.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if window_size <= 0 or (window_size % 1) != 0:
                self.loading_label.config(text=f"Error! Window size must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_waveform_long(output_name,input_audio, channel, fps, res_width, res_height, window_size, style, thickness, compression, self.update_loading_label)

        except Exception:
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class OscilloscopeWindow:
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    interpolation_values = [1,2,4,8,16,32,64]
    def __init__(self, master):
        self.master = master
        self.master.title("Oscilloscope Visualizer v0.06 by Aaron F. Bianchi")


        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.interpolation = tk.IntVar(value="1")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_combobox(self.master, "Oversampling:", self.interpolation, row=row_num, values=self.interpolation_values, tip="Will draw more points so it looks more like a continuous line.\nUses a ton of memory for high values on long songs. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        

        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        

        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            interpolation = self.interpolation.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if interpolation <= 0 or (interpolation % 1) != 0:
                self.loading_label.config(text=f"Error! Interpolation must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_oscilloscope(output_name,input_audio,fps, res_width, res_height,interpolation,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI
        
class PolarWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    interpolation_values = [1,2,4,8,16,32,64]
    def __init__(self, master):
        self.master = master
        self.master.title("Polar Visualizer v0.09 by Aaron F. Bianchi")


        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.offset = tk.DoubleVar(value=0.5)
        self.note = tk.StringVar(value="C4")
        self.interpolation = tk.IntVar(value="1")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets(self.master, "Offset:", self.offset, row=row_num, tip="Set an offset. I don't know how to explain it. Just try and see.")
        row_num += 1
        create_input_widgets(self.master, "Tuning:", self.note, row=row_num, tip="Set a note to tune the polar oscilloscope to.\nYou can enter the name of a note or its fundamental frequency in Hz.")
        row_num += 1
        create_combobox(self.master, "Oversampling:", self.interpolation, row=row_num, values=self.interpolation_values, tip="Will draw more points so it looks more like a continuous line.\nUses a ton of memory for high values on long songs. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        
        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        
        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            offset = self.offset.get()
            note = self.note.get()
            interpolation = self.interpolation.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
                
            if note.lstrip('-').replace('.', '', 1).isdigit():
                if float(note) <= 0:
                    self.loading_label.config(text=f"Error! Tuning frequency must be a positive number.")
                    error_flag = True
            else:
                if len(note) == 2:
                    if not note[0].lower() in 'abcdefg' or not note[1].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                elif len(note) == 3:
                    if not note[0].lower() in 'abcdefg' or not (note[1] == "#" or note[1] == "b") or not note[2].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                else:
                    self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                    error_flag = True    
            
            if interpolation <= 0 or (interpolation % 1) != 0:
                self.loading_label.config(text=f"Error! Interpolation must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_polar(output_name,input_audio,channel,fps, res_width, res_height, offset, note,interpolation,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI
        
class PolarStereoWindow:
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    interpolation_values = [1,2,4,8,16,32,64]
    def __init__(self, master):
        self.master = master
        self.master.title("Stereo Polar Visualizer v0.10 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.offset = tk.DoubleVar(value=0.5)
        self.note = tk.StringVar(value="C4")
        self.interpolation = tk.IntVar(value="1")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets(self.master, "Offset:", self.offset, row=row_num, tip="Set an offset. I don't know how to explain it. Just try and see.")
        row_num += 1
        create_input_widgets(self.master, "Tuning:", self.note, row=row_num, tip="Set a note to tune the polar oscilloscope to.\nYou can enter the name of a note or its fundamental frequency in Hz.")
        row_num += 1
        create_combobox(self.master, "Oversampling:", self.interpolation, row=row_num, values=self.interpolation_values, tip="Will draw more points so it looks more like a continuous line.\nUses a ton of memory for high values on long songs. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        
        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        
        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            offset = self.offset.get()
            note = self.note.get()
            interpolation = self.interpolation.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            
            if note.lstrip('-').replace('.', '', 1).isdigit():
                if float(note) <= 0:
                    self.loading_label.config(text=f"Error! Tuning frequency must be a positive number.")
                    error_flag = True
            else:
                if len(note) == 2:
                    if not note[0].lower() in 'abcdefg' or not note[1].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nC#2, Db3, E4, f#5, gb6 or a7.")
                        error_flag = True
                elif len(note) == 3:
                    if not note[0].lower() in 'abcdefg' or not (note[1] == "#" or note[1] == "b") or not note[2].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nC#2, Db3, E4, f#5, gb6 or a7.")
                        error_flag = True
                else:
                    self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nC#2, Db3, E4, f#5, gb6 or a7.")
                    error_flag = True
            
            if interpolation <= 0 or (interpolation % 1) != 0:
                self.loading_label.config(text=f"Error! Interpolation must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_polar_stereo(output_name,input_audio,fps, res_width, res_height, offset, note,interpolation,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class RecurrenceWindow:
    channel_values = ["Both (Merge to mono)", "Both (Stereo)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    def __init__(self, master):
        self.master = master
        self.master.title("Recurrence Plot Visualizer v0.10 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.note = tk.StringVar(value="C2")
        self.threshold = tk.DoubleVar(value=0.05)
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        #warning_label = tk.Label(self.master, text="WARNING: Experimental feature. If it gives you any error that you think it shouldn't give you, contact me.", fg="red")
        #warning_label.grid(row=row_num, column=0, columnspan=3, padx=(5, 5), pady=(5, 0), sticky="we")
        #row_num += 1
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. 1:1 aspect ratio is recommended. Even numbers.")
        row_num += 1
        create_input_widgets(self.master, "Tuning:", self.note, row=row_num, tip="Set a note to tune the recurrence plot to.\nYou can enter the name of a note or its fundamental frequency in Hz.")
        row_num += 1
        create_input_widgets(self.master, "Threshold:", self.threshold, row=row_num, tip="Higher values will increase the amount of white.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the whole thing one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        
        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        
        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            note = self.note.get()
            threshold = self.threshold.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
                
            if note.lstrip('-').replace('.', '', 1).isdigit():
                if float(note) <= 0:
                    self.loading_label.config(text=f"Error! Tuning frequency must be a positive number.")
                    error_flag = True
            else:
                if len(note) == 2:
                    if not note[0].lower() in 'abcdefg' or not note[1].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                elif len(note) == 3:
                    if not note[0].lower() in 'abcdefg' or not (note[1] == "#" or note[1] == "b") or not note[2].isdigit():
                        self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                        error_flag = True
                else:
                    self.loading_label.config(text=f"Error! Tuning must be written in one of the following formats:\nD#2, Db2, D2, d#2, db2 or d2.")
                    error_flag = True

            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_recurrence(output_name,input_audio,channel,fps, res_width, res_height, note, threshold, thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI

class ChladniWindow:
    channel_values = ["Both (Merge to mono)", "Both (Stereo)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    mode_values = ["Sine", "Cosine", "Tangent", "Cotangent", "Secant", "Cosecant"]
    def __init__(self, master):
        self.master = master
        self.master.title("False Chladni Plate Visualizer v0.08 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Stereo)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.mode = tk.StringVar(value="Cosine")
        self.zoom = tk.DoubleVar(value=1000)
        self.smoothing = tk.DoubleVar(value=0.05)
        self.threshold = tk.DoubleVar(value=0.5)
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0
        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_combobox(self.master, "Mode:", self.mode, row=row_num, values=self.mode_values, tip="Try these for different shapes in your video", readonly=True)
        row_num += 1
        create_input_widgets(self.master, "Zoom:", self.zoom, row=row_num, tip="Zoom in the Chladni plate.")
        row_num += 1
        create_input_widgets(self.master, "Smoothing:", self.smoothing, row=row_num, tip="The higher this value, the smoother the visualization. From 0 to 1.")
        row_num += 1
        create_input_widgets(self.master, "Threshold:", self.threshold, row=row_num, tip="Higher values will increase the amount of white.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the whole thing one pixel to the right and up.\nWill make the render slower the higher you go. Whole number")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1

        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1

        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            mode = self.mode.get()
            zoom = self.zoom.get()
            smoothing = self.smoothing.get()
            threshold = self.threshold.get()
            thickness = self.thickness.get()
            compression = self.compression.get()

            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True

            if zoom <= 0:
                self.loading_label.config(text=f"Error! Zoom amount must be a positive number.")
                error_flag = True

            if smoothing < 0 or smoothing > 1:
                self.loading_label.config(text=f"Error! Smoothing value must be from 0 to 1.")
                error_flag = True

            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True

            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_chladni(output_name,input_audio,channel,fps, res_width, res_height, mode, zoom, smoothing, threshold, thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()

    def update_loading_label(self, progress, total, text_state, text_message):
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI
        
class PoincareWindow:
    channel_values = ["Both (Merge to mono)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    interpolation_values = [1,2,4,8,16,32,64]
    def __init__(self, master):
        self.master = master
        self.master.title("Poincaré Plot Visualizer v0.01 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Merge to mono)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.delay = tk.IntVar(value=10)
        self.interpolation = tk.IntVar(value="1")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_input_widgets_num(self.master, "Delay:", self.delay, row=row_num, tip="This can modify the shape of the scribble in cool ways.\nI don't know how to explain it. Just try and see.")
        row_num += 1
        create_combobox(self.master, "Oversampling:", self.interpolation, row=row_num, values=self.interpolation_values, tip="Will draw more points so it looks more like a continuous line.\nUses a ton of memory for high values on long songs. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        
        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        
        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            delay = self.delay.get()
            interpolation = self.interpolation.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if delay <= 0:
                self.loading_label.config(text=f"Error! Delay amount must be a positive number.")
                error_flag = True
            if interpolation <= 0 or (interpolation % 1) != 0:
                self.loading_label.config(text=f"Error! Interpolation must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_poincare(output_name,input_audio,channel,fps, res_width, res_height,delay,interpolation,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI
    
class DelayEmbedWindow:
    channel_values = ["Both (Merge to mono)", "Both (Stereo)", "Left", "Right"]
    fps_values = [23.976,24,25,29.97,30,50,59.94,60,120]
    width_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    height_values = [240,360,480,540,640,720,768,960,1080,1440,1600,1920,2160]
    delay_values = [1,5,10,15,20,30,50,100]
    angle_values = [-90,-45,-30,-15,0,15,30,45,90]
    rot_speed_values = [-1,-0.5,-0.25,-0.125,0,0.125,0.25,0.5,1]
    interpolation_values = [1,2,4,8,16,32,64]
    def __init__(self, master):
        self.master = master
        self.master.title("Delay Embed Visualizer v0.05 by Aaron F. Bianchi")

        self.output_name = tk.StringVar(value="output.mp4")
        self.input_audio = tk.StringVar(value="")
        self.channel = tk.StringVar(value="Both (Stereo)")
        self.fps = tk.DoubleVar(value=60)
        self.res_width = tk.IntVar(value=720)
        self.res_height = tk.IntVar(value=720)
        self.delay1 = tk.IntVar(value=10)
        self.delay2 = tk.IntVar(value=20)
        self.alfa_p = tk.DoubleVar(value=0)
        self.alfa_s = tk.DoubleVar(value=0.25)
        self.beta_p = tk.DoubleVar(value=15)
        self.beta_s = tk.DoubleVar(value=0)
        self.interpolation = tk.IntVar(value="1")
        self.thickness = tk.IntVar(value="1")
        self.compression = tk.DoubleVar(value=0)

        row_num = 0

        create_file_input_row(self.master, "Input audio:", row=row_num, path_var=self.input_audio)
        row_num += 1
        create_file_output_row(self.master, "Output video:", row=row_num, path_var=self.output_name)
        row_num += 1
        create_combobox(self.master, "Channel:", self.channel, row=row_num, values=self.channel_values, tip=" ", readonly=True)
        row_num += 1
        create_combobox(self.master, "Frame Rate:", self.fps, row=row_num, values=self.fps_values, tip="Frames per second")
        row_num += 1
        create_combobox_dual(self.master, "Resolution:", self.res_width,"x", self.res_height, row=row_num, values=self.width_values,values2=self.height_values, tip="Width x Height. Even numbers.")
        row_num += 1
        create_combobox_dual(self.master, "Delays:", self.delay1,"&", self.delay2, row=row_num, values=self.delay_values,values2=self.delay_values, tip="This can modify the shape of the scribble in cool ways.")
        row_num += 1
        create_combobox_dual(self.master, "Rotation:", self.alfa_p,"&", self.beta_p, row=row_num, values=self.angle_values,values2=self.angle_values, tip="Initial angular positions (Pitch and Jaw) of the scribble in degrees.")
        row_num += 1
        create_combobox_dual(self.master, "Rotation Speed:", self.alfa_s,"&", self.beta_s, row=row_num, values=self.rot_speed_values,values2=self.rot_speed_values, tip="Rotation speed of the scribble in revolutions per second")
        row_num += 1
        create_combobox(self.master, "Oversampling:", self.interpolation, row=row_num, values=self.interpolation_values, tip="Will draw more points so it looks more like a continuous line.\nUses a ton of memory for high values on long songs. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Thickness:", self.thickness, row=row_num, tip="Will duplicate the curve one pixel to the right and up.\nWill make the render slower the higher you go. Whole number.")
        row_num += 1
        create_input_widgets_num(self.master, "Video Compression:", self.compression, row=row_num, tip="Constant rate factor compression. Doesn't have to be a whole number.\n- 0: No compression (~2x as fast).\n- 35: Mild compression.")
        row_num += 1
        
        self.action_button = tk.Button(self.master, text="Render video", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)
        #row_num += 1
        
        self.loading_label = tk.Label(self.master, text="Loading...", font=("Helvetica", 10), fg="blue", anchor="w", justify="left")
        self.loading_label.grid(row=row_num, column=2, padx=10, pady=5, sticky="w")
        self.loading_label.grid_remove()

    def perform_action(self):
        try:
            self.loading_label.grid()
            self.loading_label.config(fg="blue")
            self.loading_label.config(text=f"Loading...")
            self.master.update()

            output_name = self.output_name.get()
            input_audio = self.input_audio.get()
            channel = self.channel.get()
            fps = self.fps.get()
            res_width = self.res_width.get()
            res_height = self.res_height.get()
            delay1 = self.delay1.get()
            delay2 = self.delay2.get()
            alfa_p = self.alfa_p.get()
            alfa_s = self.alfa_s.get()
            beta_p = self.beta_p.get()
            beta_s = self.beta_s.get()
            interpolation = self.interpolation.get()
            thickness = self.thickness.get()
            compression = self.compression.get()
            
            error_flag = False
            if fps <= 0:
                self.loading_label.config(text=f"Error! Frame rate must be a positive number.")
                error_flag = True
            if res_width <= 0 or (res_width % 2) != 0 or res_height <= 0 or (res_height % 2) != 0:
                self.loading_label.config(text=f"Error! Resolution values must be positive even numbers.")
                error_flag = True
            if delay1 <= 0 or delay2 <= 0:
                self.loading_label.config(text=f"Error! Delay amount must be a positive number.")
                error_flag = True
            if interpolation <= 0 or (interpolation % 1) != 0:
                self.loading_label.config(text=f"Error! Interpolation must be a positive whole number.")
                error_flag = True
            if thickness <= 0 or (thickness % 1) != 0:
                self.loading_label.config(text=f"Error! Thickness must be a positive whole number.")
                error_flag = True
            if compression < 0:
                self.loading_label.config(text=f"Error! Compression must be a non-negative number.")
                error_flag = True
                
            if error_flag == True:
                self.loading_label.config(fg="Red")
                self.master.update()
            else:
                generate_delay_embed(output_name,input_audio,channel,fps, res_width, res_height,delay1,delay2, beta_p, beta_s, alfa_p, alfa_s,interpolation,thickness,compression,self.update_loading_label)

        except Exception:
            #messagebox.showerror("Error", "Invalid input. Please enter valid values.")
            self.loading_label.config(text=f"Unknown error! I checked all the fields and they seem good.\nMaybe the file doesn't exist or the sample rate is weird or smth idk.")
            self.loading_label.config(fg="Red")
            self.master.update()
            
    def update_loading_label(self, progress, total, text_state, text_message):       
        if text_state == True:
            self.loading_label.config(text=text_message)
        else:
            self.loading_label.config(text=f"Progress: Frame {progress} of {total}")
        self.master.update()  # Update the GUI
        
class HelpWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Help!!!!!!!!!!!!!111ONE")
        
        row_num = 0
        warning_label = tk.Label(self.master, text="HELP iM TRAPPED IN A VISUALIZER FACTORY IN  [R E D A C T E D]", fg="red")
        warning_label.grid(row=row_num, column=0, columnspan=3, padx=(5, 5), pady=(5, 0), sticky="we")
        
        row_num += 1
        self.action_button = tk.Button(self.master, text="Donate for rescue", command=self.perform_action)
        self.action_button.grid(row=row_num, column=1, pady=10)

        row_num += 1
        warning_label = tk.Label(self.master, text="Actually no. This program is totally free. You don't have to pay anything.\n But if you feel the need, donations are appreciated :)", fg="black")
        warning_label.grid(row=row_num, column=0, columnspan=3, padx=(5, 5), pady=(5, 0), sticky="we")

    def perform_action(self):
        webbrowser.open("https://aaron-f-bianchi.itch.io/lsao/purchase")
      
#################################################
################## MAIN WINDOW ##################
#################################################

def option_logo():
    webbrowser.open("https://aaron-f-bianchi.itch.io/lsao/purchase")

def option1():
    spectrum_window = tk.Toplevel(root)
    SpectrumWindow(spectrum_window)

def option5():
    spectrumdB_window = tk.Toplevel(root)
    SpectrumdBWindow(spectrumdB_window)

def option2():
    waveform_window = tk.Toplevel(root)
    WaveformWindow(waveform_window)
    
def option3():
    long_waveform_window = tk.Toplevel(root)
    LongWaveformWindow(long_waveform_window)
    
def option4():
    oscilloscope_window = tk.Toplevel(root)
    OscilloscopeWindow(oscilloscope_window)
    
def option6():
    polar_window = tk.Toplevel(root)
    PolarWindow(polar_window)
    
def option7():
    polar_stereo_window = tk.Toplevel(root)
    PolarStereoWindow(polar_stereo_window)
    
def option8():
    spec_balance_window = tk.Toplevel(root)
    SpecBalanceWindow(spec_balance_window) 
       
def option9():
    recurrence_window = tk.Toplevel(root)
    RecurrenceWindow(recurrence_window)
    
def option10():
    poincare_window = tk.Toplevel(root)
    PoincareWindow(poincare_window)
        
def option11():
    delay_embed_window = tk.Toplevel(root)
    DelayEmbedWindow(delay_embed_window)

def option12():
    histogram_window = tk.Toplevel(root)
    HistogramWindow(histogram_window)

def option13():
    chladni_window = tk.Toplevel(root)
    ChladniWindow(chladni_window)

#def optionhelp():
#    help_window = tk.Toplevel(root)
#    HelpWindow(help_window)
    

def update_gif(button, frames):
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

def start_gif(event, button, frames):
    if not button.gif_playing:
        button.gif_playing = True
        update_gif(button, frames)

def stop_gif(event, button):
    button.gif_playing = False 

#MAIN WINDOW
FFMPEG = ffmpeg_ubicacion()
FFPROBE = ffprobe_ubicacion()

root = tk.Tk()
root.title("LSaO Visualizer v1.02")
#root.geometry("700x600")

def load_gif(path):
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = ImageTk.PhotoImage(gif.copy())
            frames.append(frame)
            gif.seek(len(frames))
    except EOFError:
        pass
    return frames

gif1 = load_gif("resources/resized_img_spec.gif")
gif2 = load_gif("resources/resized_img_swav.gif")
gif3 = load_gif("resources/resized_img_lwav.gif")
gif4 = load_gif("resources/resized_img_osc.gif")
gif5 = load_gif("resources/resized_img_specdB.gif")
gif6 = load_gif("resources/resized_img_polar.gif")
gif7 = load_gif("resources/resized_img_polar_stereo.gif")
gif8 = load_gif("resources/resized_img_spec_balance.gif")
gif9 = load_gif("resources/resized_img_recurrence.gif")
gif10 = load_gif("resources/resized_img_poincare.gif")
gif11 = load_gif("resources/resized_img_embed2.gif")
gif12 = load_gif("resources/resized_img_histogram.gif")
gif13 = load_gif("resources/resized_img_chladni.gif")
logo = tk.PhotoImage(file="resources/lsao logotype.png")

# INITIAL TEXT
row_num = 0
initial_text = "Things to know:\n"
if os.name == 'nt':
    button_width = 110
    button_height = 104
    print("Running on Windows")
    initial_text = initial_text + "\n- The default Windows video player isn't going to play the generated videos\n  correctly. Try a better video player (VLC, for example).\n- This program uses FFmpeg."
elif os.name == 'posix':
    button_width = 90
    button_height = 104
    print("Running on Linux")
    initial_text = initial_text + "\n- You have to install FFmpeg if you haven't already."
initial_label = tk.Label(root, text=initial_text, font=("Helvetica", 10), anchor='w', justify='left')
initial_label.grid(row=row_num, column=0, columnspan=4, padx=10, pady=10, sticky="w")

# LOGO
button_logo = tk.Label(root, image=logo, text="", justify='center')
button_logo.grid(row=4, column=2, columnspan=2, padx=5, pady=5, sticky="w")
button_logo.bind("<Button-1>", lambda e: webbrowser.open("https://aaron-f-bianchi.itch.io/lsao"))

row_num += 1
col_num = 0
# OPTION 1
button_option1 = tk.Button(root, image=gif1[38], text="Linear Spectrum" , compound=tk.BOTTOM, command=option1, width=button_width, height=button_height)
button_option1.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option1.gif_playing = False
button_option1.current_frame = 38
button_option1.bind("<Enter>", lambda e: start_gif(e, button_option1, gif1))
button_option1.bind("<Leave>", lambda e: stop_gif(e, button_option1))

col_num += 1
# OPTION 5
button_option5 = tk.Button(root, image=gif5[24], text="Linear Spec. (dB)", compound=tk.BOTTOM, command=option5, width=button_width, height=button_height)
button_option5.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option5.gif_playing = False
button_option5.current_frame = 24
button_option5.bind("<Enter>", lambda e: start_gif(e, button_option5, gif5))
button_option5.bind("<Leave>", lambda e: stop_gif(e, button_option5))

col_num += 1
# OPTION 8
button_option8 = tk.Button(root, image=gif8[50], text="Spectral Balance", compound=tk.BOTTOM, command=option8, width=button_width, height=button_height)
button_option8.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option8.gif_playing = False
button_option8.current_frame = 50
button_option8.bind("<Enter>", lambda e: start_gif(e, button_option8, gif8))
button_option8.bind("<Leave>", lambda e: stop_gif(e, button_option8))

col_num += 1
# OPTION 12
button_option12 = tk.Button(root, image=gif12[5], text="Histogram", compound=tk.BOTTOM, command=option12, width=button_width, height=button_height)
button_option12.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option12.gif_playing = False
button_option12.current_frame = 5
button_option12.bind("<Enter>", lambda e: start_gif(e, button_option12, gif12))
button_option12.bind("<Leave>", lambda e: stop_gif(e, button_option12))

row_num += 1
col_num = 0

# OPTION 2
button_option2 = tk.Button(root, image=gif2[53], text="Short Waveform", compound=tk.BOTTOM, command=option2, width=button_width, height=button_height)
button_option2.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option2.gif_playing = False
button_option2.current_frame = 53
button_option2.bind("<Enter>", lambda e: start_gif(e, button_option2, gif2))
button_option2.bind("<Leave>", lambda e: stop_gif(e, button_option2))

col_num += 1
# OPTION 3
button_option3 = tk.Button(root, image=gif3[30], text="Long Waveform", compound=tk.BOTTOM, command=option3, width=button_width, height=button_height)
button_option3.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option3.gif_playing = False
button_option3.current_frame = 30
button_option3.bind("<Enter>", lambda e: start_gif(e, button_option3, gif3))
button_option3.bind("<Leave>", lambda e: stop_gif(e, button_option3))

col_num += 1
# OPTION 9
button_option9 = tk.Button(root, image=gif9[111], text="Recurrence", compound=tk.BOTTOM, command=option9, width=button_width, height=button_height)
button_option9.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option9.gif_playing = False
button_option9.current_frame = 111
button_option9.bind("<Enter>", lambda e: start_gif(e, button_option9, gif9))
button_option9.bind("<Leave>", lambda e: stop_gif(e, button_option9))

col_num += 1
# OPTION 4
button_option4 = tk.Button(root, image=gif4[109], text="Oscilloscope", compound=tk.BOTTOM, command=option4, width=button_width, height=button_height)
button_option4.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option4.gif_playing = False
button_option4.current_frame = 109
button_option4.bind("<Enter>", lambda e: start_gif(e, button_option4, gif4))
button_option4.bind("<Leave>", lambda e: stop_gif(e, button_option4))

row_num += 1
col_num = 0

# OPTION 6
button_option6 = tk.Button(root, image=gif6[228], text="Polar (Mono)", compound=tk.BOTTOM, command=option6, width=button_width, height=button_height)
button_option6.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option6.gif_playing = False
button_option6.current_frame = 228
button_option6.bind("<Enter>", lambda e: start_gif(e, button_option6, gif6))
button_option6.bind("<Leave>", lambda e: stop_gif(e, button_option6))

col_num += 1
# OPTION 7
button_option7 = tk.Button(root, image=gif7[575], text="Polar (Stereo)", compound=tk.BOTTOM, command=option7, width=button_width, height=button_height)
button_option7.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option7.gif_playing = False
button_option7.current_frame = 575
button_option7.bind("<Enter>", lambda e: start_gif(e, button_option7, gif7))
button_option7.bind("<Leave>", lambda e: stop_gif(e, button_option7))

col_num += 1
# OPTION 10
button_option10 = tk.Button(root, image=gif10[41], text="Poincaré", compound=tk.BOTTOM, command=option10, width=button_width, height=button_height)
button_option10.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option10.gif_playing = False
button_option10.current_frame = 41
button_option10.bind("<Enter>", lambda e: start_gif(e, button_option10, gif10))
button_option10.bind("<Leave>", lambda e: stop_gif(e, button_option10))

col_num += 1
# OPTION 11
button_option11 = tk.Button(root, image=gif11[97], text="Delay Embed", compound=tk.BOTTOM, command=option11, width=button_width, height=button_height)
button_option11.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option11.gif_playing = False
button_option11.current_frame = 97
button_option11.bind("<Enter>", lambda e: start_gif(e, button_option11, gif11))
button_option11.bind("<Leave>", lambda e: stop_gif(e, button_option11))

row_num += 1
col_num = 0

# OPTION 13
button_option13 = tk.Button(root, image=gif13[11], text="Chladni", compound=tk.BOTTOM, command=option13, width=button_width, height=button_height)
button_option13.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="we")
button_option13.gif_playing = False
button_option13.current_frame = 11
button_option13.bind("<Enter>", lambda e: start_gif(e, button_option13, gif13))
button_option13.bind("<Leave>", lambda e: stop_gif(e, button_option13))

# col_num += 1
# # Button for option 12
# button_option12 = tk.Button(root, image=gif12[0], text="H E L P", compound=tk.BOTTOM, command=option12, width=130, height=130)
# button_option12.grid(row=row_num, column=col_num, padx=0, pady=0, sticky="w")
# button_option12.gif_playing = False
# button_option12.current_frame = 0
# button_option12.bind("<Enter>", lambda e: start_gif(e, button_option12, gif12))
# button_option12.bind("<Leave>", lambda e: stop_gif(e, button_option12))

row_num += 1
# Credits label
credits_label = tk.Label(root, text="© 2025 Aaron F. Bianchi", font=("Helvetica", 10), fg="blue", justify='center')
credits_label.grid(row=row_num, column=0, columnspan=5, pady=5, sticky="ew") 
credits_label.bind("<Button-1>", lambda e: webbrowser.open("https://aaronfbianchi.github.io/"))

root.mainloop()
