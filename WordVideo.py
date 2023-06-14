import glob
import os
import math
#Lrbl2p
# from moviepy.editor import *
import cv2
from pathlib import Path
import ROI
import TPE
import Features
import Gabor
import traceback
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import datetime


##Using generator to process files

def Frame(detector, predictor,VideoPath,FramePath,MouthPath,GaborPath,SheetPath,FeaturesPath):

    if not os.path.exists(FramePath):
        os.mkdir(FramePath)
    #obtain the individual words

    def generate_speakers():
        for m in range(0, 1):
            yield m

    def generate_videos(video):
        for v in video:
            yield v

    speaker_generator = generate_speakers()
    video_generator = generate_videos(glob.glob(VideoPath, recursive=True))

    for m in speaker_generator:
        for v in video_generator:
            (filepath, tempfilename) = os.path.split(v)
            maindir, videodir = os.path.split(filepath)
            (video_shotname, extension) = os.path.splitext(tempfilename)
            folder_name = videodir + '_' + video_shotname
            Path = os.path.join(FramePath, folder_name)
            if not os.path.exists(Path):
                os.mkdir(Path)

            cap = cv2.VideoCapture(v)
            fps = cap.get(5)
            totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            i = 0
            while True:
                ret, frames = cap.read()
                if ret == True:
                    i = i + 1
                    path = Path + '/'
                    picturepath = path + str('%02d' % i) + '.jpg'
                    print(picturepath)
                    cv2.imwrite(picturepath, frames)
                        
                    try:
                        ROIpath, mouth_centroid_x, mouth_centroid_y, ROI_mouth, widthG, heightG = ROI.rect1(
                                        detector, predictor, i, folder_name, picturepath, MouthPath, GaborPath, SheetPath, FeaturesPath)
                    except Exception as e:
                        traceback.print_exc()
                        continue
                        
                    global HGamma, HKernelSize, HSig, HWavelength
                    while True:
                        try:
                            if i == 1:
                                HGamma, HKernelSize, HSig, HWavelength = TPE.TPE(picturepath, mouth_centroid_x,
                                                                                 mouth_centroid_y, ROI_mouth,
                                                                                 widthG, heightG)
                            Gabor_Path = Gabor.Gabor_h(HGamma, HKernelSize, HSig, HWavelength, i, ROIpath,
                                                       folder_name, GaborPath)
                            Features.Features(mouth_centroid_x, mouth_centroid_y, i, folder_name,
                                              Gabor_Path, SheetPath, FeaturesPath)
                        except Exception as e:
                            print("Try again")
                            traceback.print_exc()
                            continue
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break