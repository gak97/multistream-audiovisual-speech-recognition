import glob
import cv2
import os
import ROI
import Gabor
import Features
import Frame
import dlib
import time

a=time.time()

#  Dlib detector location
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

Video_path = r'../Test20May2022/bbae2a.mpg' # path of video (for a group of video, you can use /*.mpg)
PicturePath = '../Test20May2022/Picture/'# path to store pictures
MouthPath = '../Test20May2022/mouth/'  # path to store mouth
GaborPath = '../Test20May2022/Gabor/'#path to store Gabor features
SheetPath = '../Test20May2022/Sheet/' # path to storSheetPath
FeaturesPath = '../Test20May2022/Features/'  # path to store sheets
Frame.Frame(detector,predictor,Video_path,PicturePath,MouthPath,GaborPath,SheetPath,FeaturesPath)
b=time.time()
print("Time = ",b-a)