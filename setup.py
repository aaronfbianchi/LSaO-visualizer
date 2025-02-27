from cx_Freeze import setup, Executable
#python setup.py build

setup(
    name="LSaO Visualizer",
    #version="0.55",
    #description="Linear Spectrum and Osciloscope Visualizer Generator",
    executables=[Executable("main.py")]
)
