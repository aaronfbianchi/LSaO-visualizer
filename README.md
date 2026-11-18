# LSaO Visualiser
by [Aaron F. Bianchi](https://aaronfbianchi.github.io/) 

Linear Spectrum and Oscilloscope Visualizer is an extremely fast audio visualization tool

This tool can export videos of a linear spectrum, tuned short waveform, long waveform, X/Y oscilloscope, vectorscope, polar oscilloscope, recurrence plot, Poincaré plot, Chladni plate simulator or delay embedding plot of a song. It supports multiple audio and video formats.

Coded to look as **violent**, **responsive**, **snappy** and **rough** as possible. It draws a white visualizer over a black background for further processing with a video editor of your preference.

Linear Spectrum:

![Example Spectrum GIF](img/example-spectrum.gif) 


Short Waveform:

![Example Short Waveform GIF](img/example-short-waveform.gif) 


Linear Spectrum (dB):

![Example Spectrum (dB) GIF](img/example-spectrum-dB.gif) 


Spectrogram:

![Example Spectrogram GIF](img/example-spectrogram.gif) 


Long Waveform:

![Example Long Waveform GIF](img/example-long-waveform.gif) 


Stereo Oscilloscope:

![Example Stereo Oscilloscope GIF](img/example-osc-github.gif) 


Polar Vectorscope:

![Example Vectorscope GIF](img/example-vectorscope.gif) 


Mono Polar Oscilloscope:

![Example Mono Polar Oscilloscope GIF](img/example-polar-github.gif) 


Stereo Polar Oscilloscope (one audio source and various tunings):

![Example Stereo Polar Oscilloscope GIF](img/example-polar-stereo.gif) 


Mono Polar Oscilloscope:

![Example Mono Polar Oscilloscope 2 GIF](img/example-polar-death.gif) 


Stereo Polar Oscilloscope:

![Example Stereo Polar Oscilloscope 2 GIF](img/example-polar-stereo-death.gif)


Linear Spectral Balance:

![Example Linear Spectral Balance GIF](img/example-balance-github.gif)


Recurrence Plot:

![Example Recurrence GIF](img/example-recurrence.gif)


Poincaré Plot

![Example Poincaré GIF](img/example-poincare.gif)


Delay Embedding Plot:

![Example Delay Embedding GIF](img/example-embed.gif)


Histogram Visualizer:

![Example Histogram GIF](img/example-histogram.gif)


False Chladni Plate Visualizer:

![Example False Chladni GIF](img/example-false-chladni.gif)


Chladni Plate Visualizer:

![Example Chladni GIF](img/example-chladni.gif)


Envelope Visualizer:

![Example Envelope GIF](img/example-envelope.gif)


Bunch of demo videos
---------------------
* [Hellhacker](https://www.youtube.com/watch?v=upkUpTIws48) by Aaron F. Bianchi
* [Slaying With Portals](https://www.youtube.com/watch?v=IIGqghktYas) by Aaron F. Bianchi
* [The Forbidden Dance](https://www.youtube.com/watch?v=qKTOINiTxGw) by The Hamster Alliance
* [Nailgun](https://www.youtube.com/watch?v=buWPKEcAkw8) by Aaron F. Bianchi
* [Deathmatch EP](https://www.youtube.com/watch?v=_H94n6kc204) by Aaron F. Bianchi
* [Blip](https://youtu.be/unIQeniWOp0) by The Hamster Alliance
* [Death Itself](https://youtu.be/11eqIOSgt0w?si=JjFL0j3CZkbPQk2i) by Aaron F. Bianchi

Why?
---------------------
This small project started due to the omnipresence of logarithmic spectrum visualizers and the apparent absence of linear spectrum visualizers (apart from the one in After Effects, but who wants to install an Adobe product anyway).

Download
---------------------
You can download it [here](https://codeberg.org/aaronfbianchi/LSaO-visualizer/releases).

Installation
---------------------
You don't need to install it. Just double click the executable.

Live Audio Visualization
---------------------
**For Linux:**

You can select which audio stream you want to visualize with Pavucontrol. Just set that source as fallback in the "Input Devices" tab. 

- Set as fallback "Monitor" to visualize Desktop Audio.
- Set as fallback Internal Microphone or your audio interface for actual audio inputs.

<img width="865" height="369" alt="image" src="https://github.com/user-attachments/assets/cde18d65-39cf-44ea-a967-bb571e5beabf" />

**For Windows:**

- Search for "Sound settings"

<img width="233" height="213" alt="image" src="https://github.com/user-attachments/assets/84020ae1-d011-44eb-926b-0ac6da64f6e2" />

- Advanced -> More sound settings

<img width="326" height="347" alt="image" src="https://github.com/user-attachments/assets/5796f39b-95ee-4bb8-85ff-dba6ce8dfe19" />

- For Desktop Audio, you can select which audio stream you want to visualize in the "Playback" tab by setting it as default.

<img width="400" height="454" alt="image" src="https://github.com/user-attachments/assets/84aac436-b8ad-4ed5-9457-022698813853" />

- For live microphone input (or any other audio source like an audio interface), go to the "Recording" tab and double click on the device you want to use.

<img width="395" height="451" alt="image" src="https://github.com/user-attachments/assets/8f99f074-e34d-4777-98b8-b1e843fee6b8" />

- In the "Listen" tab, check "Listen to this Device" and then press OK/Apply.

<img width="398" height="445" alt="image" src="https://github.com/user-attachments/assets/383b1178-7706-4f27-9a92-48b7b86db1e4" />


Things to know
---------------------
* The Windows releases of this software bundle FFmpeg (https://ffmpeg.org/). Nevertheless, if for any reason it's missing, it will try to use the FFmpeg in PATH if already installed. FFmpeg is licensed under the GNU General Public License version 3 (GPL v3). Prebuilt binaries used in this project are from Gyan.dev (https://www.gyan.dev/ffmpeg/builds/). The FFmpeg source code is available at https://ffmpeg.org/download.html#source


Tips if using on Windows
-------------------------
* The default Windows video player is not gonna play the exported videos correctly. Try another video player instead (Like VLC, for example). You'll be able to use the exported videos in any video editing software just fine, though.


Running from source
---------------------
Version 3.00 uses 'uv' as the library manager. To run this program, clone it and execute 'uv run src/main.py' in the command line.

Aditionally, if you want to create an executable from source, these libraries have worked for me:
* cx-freeze==8.7.1 (For Linux)
* pyinstaller>=6.22.3 (For Windows)

For creating an executable from Linux:

    uv run cxfreeze build_exe

For creating an executable from Windows:

    uv run pyinstaller --onefile --noupx --add-data resources/*:resources src/main.py
