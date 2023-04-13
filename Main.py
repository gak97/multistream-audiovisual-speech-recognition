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
a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#VideoPath = r'../Test20May2022/*.mpg'
#VideoPath ='../Test20May2022/bbae2a.mpg'
VideoPath ='../Test20May2022/huo3.mp4'
Frame = '../Test20May2022/Picture/'# path to store pictures
MouthPath = '../Test20May2022/mouth/'  # path to store mouth
GaborPath = '../Test20May2022/Gabor/'#path to store Gabor features
SheetPath = '../Test20May2022/Sheet/' # path to storSheetPath
FeaturesPath = '../Test20May2022/Features/'  # path to store sheets


WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)


b=time.time()
print("Time = ",b-a)