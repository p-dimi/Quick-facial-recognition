# Face-Recognition
Easy to use and utilize face recognition module based on dlib

Utilizes open source dlib, opencv and numpy libraries.
Requirements in the requirement.txt file.


Simply import, initialize, and pass a picture of a face you wish to recognize.
Module will compare faces stored in a local folder (a known faces folder) vs the face you pass it, and perform the facial comparison for recognition.
Module initialization might take some time, as it initially processes all faces in local folder into corresponding vector face representations, for faster real-time comparisons. The slower initialization will allow for faster runtime.


Class returns the person recognized (using the name of the picture file it is most resembling), and linear distance score.
Class also displays a window with the image you are trying to recognize, and the image it is most similar with from your known faces folder.

### The class takes the following variables upon initializations:

#### recognition_threshold
The linear distance score that the face similarity must pass (be under). A higher value is more permitting, a lower value is more restricting. 

The value is 0.6 by default.

#### size_modifier
A float value representing by how much to scale all images for faster performance. A lower value will make runtime faster but be less accurate. A higher value will make runtime slower but more accurate. For example - a value of 0.75 means resizing the images to 75% of their original scale.

The value is 0.75 by default. It is advised not to set it higher than 1.0, and no lower than 0.5.

#### known_faces
The folder path of your known faces. This is the folder containing the pictures you want to use for comparison when performing the facial recognition.

The default path is 'known_faces/'

It is advised to keep image names in the folder simple and in JPG or PNG format. For instance: "josh.jpg", "karen.jpg", "alex.jpg" etc

#### predictor_path
The full path of your dlib landmarks predictor model.

The default path is 'dlib_models/shape_predictor_68_face_landmarks.dat'

#### recognition_model_path
The full path of your dlib face recognition model.

The default path is 'dlib_models/dlib_face_recognition_resnet_model_v1.dat'

#### display_window
Boolean value for whether or not to display the images (the image fed to the class and the picture of the person recognized from the known faces folder) side by side, or not display any images during the process.

The default value is True


# Example use
```
# cv2 for video capture from webcam
import cv2

# keyboard module to be able to quit by pressing "q"
from keyboard import is_pressed as key

# Import the face_recognizer class from the face_recog module
from face_recog import face_recognizer

# initialize the face recognizer with all default init variables
f_r = face_recognizer()

# set to capture webcam live feed
capture = cv2.VideoCapture(0)

while True:
    # get frame from webcam
    ret, frame = capture.read()

    # perform face recognition on the frame
    person, similarity = f_r.recognize(frame)
    
    # print the recognized person's name and linear distance value
    print(person)
    print(similarity)
    
    # break by pressing "q"
    if key('q'):
        break

capture.release()
```
