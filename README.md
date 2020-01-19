# Face-Recognition
Easy to use and utilize face recognition module based on dlib.

Example use code found below.

## Requirements
Requirements in the requirement.txt file.

Utilizes open source dlib, opencv and numpy libraries.
The dlib face recognition model (pretrained resnet) is included in the repo in the dlib_models folder.
#### You will also need the dlib 68 landmarks predictor model. Download it here:

https://github.com/italojs/facial-landmarks-recognition-/blob/master/shape_predictor_68_face_landmarks.dat

#### Make sure to save the downloaded landmark predictor model into the dlib_models folder (or whichever folder you intend to store the dlib models in)

## Downloading
Simply download the repo and place it in the folder of whichever project you'd like to use it on. Feel free to store the dlib models and known faces in a different directory.

## Using
Simply import, initialize, and pass a picture of a face you wish to recognize.

Store your face database in the known_faces folder (or any other directory you wish to use). It is advised the faces are at least 256x256 pixels in size, and that the pictures are named in accordance with the person's name.

The module will compare faces stored in a local folder (a known faces folder) versus the face you pass it, and perform the facial comparison for recognition.

Module initialization might take some time, as it initially processes all faces in local folder into corresponding vector face representations, for faster real-time comparisons. The slower initialization will allow for faster runtime.


Class returns a dictionary, with the keys representing the recognized person(s) (using the name of the picture file it is most resembling), bounding box position in the image, and linear distance score.

### The class takes the following variables upon initializations:

#### recognition_threshold
The linear distance score that the face similarity must pass (be under). A higher value is more permitting, a lower value is more restricting. 

The value is 0.5 by default.

#### size_modifier
A float value representing by how much to scale all images for faster performance. A lower value will make runtime faster but be less accurate. A higher value will make runtime slower but more accurate. For example - a value of 0.75 means resizing the images to 75% of their original scale.

The value is 0.75 by default. It is advised not to set it higher than 1.0, and no lower than 0.5.

#### init_size_modifier
Like size_modifier but specifically for the pictures in your known_faces folder. Makes for slightly faster initialization.

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
    faces = f_r.recognize(frame)

    # 
    for face in faces:
        try:
            top_left = faces[face][0][0]

            bottom_right = faces[face][0][1]
            
            # show the bounding box, name, and confidence of the person found
            cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 2)

            cv2.putText(frame, face +' '+ str(round(faces[face][1], 2)), (top_left[0],top_left[1]+30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        except:
            # in the case there's nothing to show, just pass
            pass
    
    # show your frame with the person(s) detected
    cv2.imshow('display window', frame)
    
    # break by pressing ESC
    k = cv2.waitKey(1)
    if k == 27:
        break
        
cv2.destroyAllWindows()
capture.release()
```
