# FaceRecognition-C++
A C++ app to face recognition using opencv and dlib

# Dependences
- Compile and configure Dlib to visual studio. Read the documentation on <a href="http://dlib.net/compile.html">Dlib Documentation</a>/br>
- Compile and configure Opencv to visual studio.

# Train

To train your own dataset:
- Create folders with the name equal the label that you want recognize the person. In there, add photos of the people with only one face per photo.
- Execute the function 'trainModel()' located in Train.h file.

# Test

To test the algorithm:
- Execute the webcamTest function to use webCam or imgTest to use an image of your system.


## ONLY USE IMAGES .BMP, BOTH IN TRAIN AND TEST. IMAGENS IN OTHER FORMAT DON'T WORK.
  
