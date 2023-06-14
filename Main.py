import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import TPE
import Gabor
import Features

# import sys

# def ignore_unraisablehook(exc_type, exc_value, exc_traceback, err_msg, obj):
#     pass

# sys.unraisablehook = ignore_unraisablehook

a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#VideoPath = r'May2023/*.mpg'
# VideoPath ='May2023/bbae2a.mpg'
# VideoPath ='May2023/huo3.mp4'
VideoPath = "A:/MSc/Dissertation (CS958)/lrs2_v1/mvlrs_v1/main/**/*.mp4"

FramePath = '29May2023/Picture/'# path to store pictures
MouthPath = '29May2023/mouth/'  # path to store mouth
GaborPath = '29May2023/Gabor/'#path to store Gabor features
SheetPath = '29May2023/Sheet/' # path to storSheetPath
FeaturesPath = '29May2023/Features/'  # path to store sheets


WordVideo.Frame(detector,predictor,VideoPath,FramePath,MouthPath,GaborPath,SheetPath,FeaturesPath)


b=time.time()
print("Time = ",b-a)