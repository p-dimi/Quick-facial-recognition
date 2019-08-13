# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:35:29 2018

@author: Dima's_Monster
"""
import dlib
import cv2
import numpy as np
import os

class face_recognizer:
    def __init__(self, recognition_threshold = 0.5, size_modifier = 0.75, known_faces = 'known_faces/', predictor_path = 'dlib_models/shape_predictor_68_face_landmarks.dat', recognition_model_path = 'dlib_models/dlib_face_recognition_resnet_model_v1.dat', display_window = True):
        # size modifier for faster processing
        self.size_modifier = size_modifier
        # recognition threshold - how similar must the faces be to be considered the same person
        self.threshold = recognition_threshold
        
        self.detector = dlib.get_frontal_face_detector() 
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        self.face_recognizer = dlib.face_recognition_model_v1(recognition_model_path)
        self.known_faces_dir = known_faces
        self.known_face_files = os.listdir(known_faces)
        self.known_faces = [name[:-3] for name in self.known_face_files]
        
        # initialize all face vectors
        self.face_vector_dict = {}
        self.initial_vector_processing()
        
        self.display_window = display_window
        if display_window:
            self.window = dlib.image_window()
            self.window_rec = dlib.image_window()
            
    def invert_channels(self, img):
        img_n = img.copy()
        img_n[:,:,0] = img[:,:,2]
        img_n[:,:,2] = img[:,:,0]
        return img_n
    
    def process_faces(self, img):
        detected = self.detector(img, 1)
        d = detected[0]
        shape = self.shape_predictor(img, d)
        return d, shape
    
    def initial_vector_processing(self):
        # initialize recognizer for faster real time performance
        for i in range(len(self.known_faces)):
            pth = ''.join([self.known_faces_dir, self.known_face_files[i]])
            compared_face_img = cv2.resize(cv2.imread(pth), (0,0), fx=self.size_modifier, fy=self.size_modifier)
            compared_face_img = self.invert_channels(compared_face_img)
            
            person_name = self.known_faces[i]
            
            compared_d, compared_shape = self.process_faces(compared_face_img)
            
            compared_face_vector = self.face_recognizer.compute_face_descriptor(compared_face_img, compared_shape)
            
            self.face_vector_dict[person_name] = (compared_face_vector, pth)
         
    def compare_with_knowns(self, face_vector):
        for name in self.face_vector_dict:
            compared_face_vector = self.face_vector_dict[name][0]
            similarity = np.linalg.norm(np.asarray(face_vector) - np.asarray(compared_face_vector))
            if similarity <= 1 - self.threshold:
                # returns the known person image file path, name of the person recognized and the percentage of similarity
                return self.face_vector_dict[name][1], name, similarity
        # if nothing is found
        print('No similar person found')
        return None, None, None
        
    def recognize(self, img):
        # resize image for faster processing
        img = cv2.resize(img, (0,0), fx=self.size_modifier, fy=self.size_modifier)
        
        # turn img from bgr to rgb
        img = self.invert_channels(img)
        
        # process faces
        try:
            d, shape = self.process_faces(img)
            # get 128D point vector
            face_vector = self.face_recognizer.compute_face_descriptor(img, shape)
            # process known faces, and compare similarities
            known_image_path, name, similarity = self.compare_with_knowns(face_vector)
            
            if known_image_path == None and name == None and similarity == None:
                return None, None
            
            if self.display_window:
                # clear out window to display new frame
                self.window.clear_overlay()
                self.window.set_image(img)
                
                #pth = ''.join([self.known_faces_dir, self.known_face_files[index]])
                compared_face_img = self.invert_channels(cv2.imread(known_image_path))
                
                self.window_rec.clear_overlay()
                self.window_rec.set_image(compared_face_img)
                
                # draw face/s
                self.window.clear_overlay()
                self.window.add_overlay(d)
                self.window.add_overlay(shape)
            
            # return the name of the person and percentage of similarity
            return name, similarity
        except:
            print('No face detected in frame')
            return None, None