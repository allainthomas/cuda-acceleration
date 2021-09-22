# cuda-acceleration
This repository proposes several scripts in order to test the acceleration received by the use of CUDA accelerated libraries in Python.

The goal is to compare the processing speed of various operations either called by Numpy or Cupy.
Cupy is a kind of GPU accelerated version of Numpy. And as such, most of the operations available via Numpy, have a counterpart with Cupy.
As the acceleration from Cupy comes from the use of CUDA, it is only available to compatible Nvidia GPU. It is possible to check if CUDA is properly installed with the command nvidia-smi in the terminal.

The scripts covers various mathematical operations :
- Creation of arrays
- FFT
- Sum of arrays
- Multiplication of arrays
- Standard deviation
- Element wise 
- Slicing
- SVD

The test consists in running the operations with 3 arrays of shapes (1000,1000), (5000,5000) and (10000,10000), with Numpy and with Cupy (6 tests per scripts).
It is different for the SVD script as using a shape of (10000,10000) requires to much memory.

Everything was tested here with a Intel Xeon 2.20 GHz CPU and a Nvidia Tesla K80 GPU.
Every test script contains the result from the test on my side, with the indicated execution time (vary on array shape and hardware)

# Requirements
To use this repository, you will need :
- numpy
- cupy
- pytest-benchmark
To install Cupy, make sure to have a Nvidia GPU and to have CUDA installed as well as the drivers properly installed.

# How to use
Each test is associated to a script file. These files are stored in the test_scripts/ folder.
To run them, the command use is :
python3 -m pytest test_scripts/test_xxxxx.py