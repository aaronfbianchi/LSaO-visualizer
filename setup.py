from cx_Freeze import setup, Executable
#python3 setup.py build

setup(
    name="LSaO Visualizer",
    description="Extremely Violent and Fast Visualization Tool",
    executables=[Executable("main.py", target_name="LSaO Visualizer")],
)

