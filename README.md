# LSaO Visualiser
**L**inear **S**pectrum **a**nd **O**scilloscope Visualizer is an extremely fast audio visualization tool

by [Aaron F. Bianchi](mailto:aaronf.bianchi@gmail.com) 

This tool can export videos of a linear spectrum, tuned short waveform, long waveform or stereo oscilloscope of a song.  It supports multiple audio and video formats.

Written to output videos that look as **violent**, **responsive**, **snappy** and **rough** as possible. It draws a white linear spectrum, waveform or oscilloscope over a black background for further processing with a video editor of your preference.

![example-spectrum](https://github.com/aaronfbianchi/LSaO/blob/main/img/example-spectrum.gif "example-spectrum")
![example-short-waveform](https://github.com/aaronfbianchi/LSaO/blob/main/img/example-short-waveform.gif "example-short-waveform")

Bunch of demo videos
---------------------
* [Hellhacker](https://www.youtube.com/watch?v=upkUpTIws48) by Aaron F. Bianchi
* [Slaying With Portals](https://www.youtube.com/watch?v=IIGqghktYas) by Aaron F. Bianchi
* [The Forbidden Dance](https://www.youtube.com/watch?v=qKTOINiTxGw) by The Hamster Alliance
* [Nailgun](https://www.youtube.com/watch?v=buWPKEcAkw8) by Aaron F. Bianchi
* [Deathmatch EP](https://www.youtube.com/watch?v=_H94n6kc204) by Aaron F. Bianchi

Why?
---------------------
This small project started due to the omnipresence of logarithmic spectrum visualizers and the apparent absence of linear spectrum visualizers (apart from the one in After Effects). Once I made this, I decided to also include the other three types of visualizers.

Usage
---------------------
* Place your audio file in the same folder the executable is.
* The video will be exported to the same folder. If there's another one with the same name as the output, it will overwrite it without asking.
* FFmpeg is required.
* The oscilloscope generator only works with stereo files as it wouldn't make any sense to visualize a mono file with a stereo oscilloscope.

Instalation
---------------------
You don't need to install it. Just double click the executable.

Building from source
---------------------
Being a python program, you'll need some libraries: (You'll probably have most of them already just by having python anyway)
* tkinter
* numpy
* scipy
* subprocess
* os
* webbrowser
