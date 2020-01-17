# FaceRecognition-C++
A C++ app to face recognition using opencv and dlib

# Dependences
- Build and configure Dlib for visual studio. Read the documentation on <a href="http://dlib.net/compile.html">Dlib Documentation</a></br>
- Build and configure Opencv for visual studio. <a href="https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html">This tutorial maybe help</a></br>

# Training

To train your own dataset:
- Create folders with the name equal the label that you want recognize the person. In there, add photos of the people with only one face per photo.
- Execute the function 'trainModel()' located in Train.h file.

# Test

To test the algorithm:
- Execute the webcamTest function to use webCam or imgTest to use an image of your system.


## ONLY USE IMAGES .BMP, BOTH IN TRAINING AND TEST. IMAGES IN OTHER FORMATS DON'T WORK.
  
