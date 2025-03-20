# LSaO Visualiser
by [Aaron F. Bianchi](https://aaronfbianchi.github.io/) 

Linear Spectrum and Oscilloscope Visualizer is an extremely fast audio visualization tool

This tool can export videos of a linear spectrum, tuned short waveform, long waveform, X/Y oscilloscope, polar oscilloscope or recurrence plot of a song.  It supports multiple audio and video formats.

Coded to look as **violent**, **responsive**, **snappy** and **rough** as possible. It draws a white linear spectrum, waveform or oscilloscope over a black background for further processing with a video editor of your preference.

Linear Spectrum:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-spectrum.gif"
       alt="Example Spectrum GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
</div>
Short Waveform:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-short-waveform.gif"
       alt="Example Short Waveform GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
</div>
Linear Spectrum (dB):
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-spectrum-dB.gif"
       alt="Example Spectrum dB GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
</div>
Long Waveform:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-long-waveform.gif"
       alt="Example Long Waveform GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
</div>
Stereo Oscilloscope and Mono Polar Oscilloscope:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-osc-github.gif"
       alt="Example Stereo Oscilloscope GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-polar-github.gif"
       alt="Example Polar Oscilloscope GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
</div>
Stereo Polar Oscilloscope (various tunings):
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-polar-stereo.gif"
       alt="Example Stereo Polar Oscilloscope GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
</div>
Mono Polar Oscilloscope and Stereo Polar Oscilloscope:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-polar-death.gif"
       alt="Example Polar Oscilloscope (Violent) GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-polar-stereo-death.gif"
       alt="Example Stereo Polar Oscilloscope (Violent) GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
</div>
Linear Spectral Balance and Recurrence Plot:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-balance-github.gif"
       alt="Example Spectral Balance GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-recurrence.gif"
       alt="Example Recurrence Plot GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
</div>
Poincaré Plot and Delay Embedding Plot:
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-poincare.gif"
       alt="Example Poincaré Plot GIF"
       style="max-width: 100%; height: auto; margin: 10px;">
  <img src="https://github.com/aaronfbianchi/LSaO-visualizer/blob/main/img/example-embed.gif"
       alt="Example Delay Embedding Plot GIF"
       style="max-width: 100%; height: auto; margin: 0px;">
</div>


Bunch of demo videos
---------------------
* [Hellhacker](https://www.youtube.com/watch?v=upkUpTIws48) by Aaron F. Bianchi
* [Slaying With Portals](https://www.youtube.com/watch?v=IIGqghktYas) by Aaron F. Bianchi
* [The Forbidden Dance](https://www.youtube.com/watch?v=qKTOINiTxGw) by The Hamster Alliance
* [Nailgun](https://www.youtube.com/watch?v=buWPKEcAkw8) by Aaron F. Bianchi
* [Deathmatch EP](https://www.youtube.com/watch?v=_H94n6kc204) by Aaron F. Bianchi
* [The Blip](https://www.youtube.com/watch?v=6q7hULl50wA) by The Hamster Alliance
* [Death Itself](https://youtu.be/11eqIOSgt0w?si=JjFL0j3CZkbPQk2i) by Aaron F. Bianchi

Why?
---------------------
This small project started due to the omnipresence of logarithmic spectrum visualizers and the apparent absence of linear spectrum visualizers (apart from the one in After Effects, but who wants to install an Adobe product anyway).

Download
---------------------
You can download it [here](https://github.com/aaronfbianchi/LSaO-visualizer/releases).

Installation
---------------------
You don't need to install it. Just double click the executable.

Usage
---------------------
* Place your audio file in the same folder the executable is.
* The video will be exported to the same folder. If there's another one with the same name as the output, it will overwrite it without asking.
* FFmpeg is required.
* The oscilloscope generator only works with stereo files as it wouldn't make any sense to visualize a mono file with a stereo oscilloscope.

Tips if using on Windows
-------------------------
* You need to install FFmpeg manually and add it to PATH. Tthis is NOT just downloading a random and of dubious origin "ffmpeg.exe" file and pasting it into the root folder of this program. You can follow [this tutorial](https://phoenixnap.com/kb/ffmpeg-windows) on how to properly do it.
* The default Windows video player is not gonna play the exported videos correctly. Try VLC instead. (You'll be able to use the exported videos in any video editing software just fine, though)


Running from source
---------------------
Being a Python 3.11.2 program, you'll need some libraries, which you can see in the "requirements.txt" file. To install them, run these commands.

For Linux:

    # For setting up your virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # For installing the libraries
    pip install -r requirements.txt

For Windows:

    # For setting up your virtual environment
    python3 -m venv venv
    venv\Scripts\activate

    # For installing the libraries
    pip install -r requirements.txt

Aditionally, if you want to create an executable from source, these libraries have worked for me:
* cx_Freeze v6.15.14 (For Linux)
* pyinstaller v6.3.0 (For Windows)

For creating executable from Linux:

    python setup.py build

For creating executable from Windows:

    pyinstaller --onefile --console main.py

