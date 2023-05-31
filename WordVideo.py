import glob

from moviepy.editor import *
import cv2
from pathlib import Path
import ROI
import TPE
import Features
import Gabor
import traceback
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# import datetime
# import time
# import subprocess
# import shlex


def Frame(detector, predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath):

    if not os.path.exists(Frame):
        os.mkdir(Frame)
    #obtain the individual words


    video=glob.glob(VideoPath, recursive=True)
    for m in range(0,1):
            # print(video[m])
        for v in video:  # path of videos
            # print(v)
            (filepath, tempfilename) = os.path.split(v)
            # print(filepath, tempfilename)
            maindir, videodir = os.path.split(filepath)
            # print(videodir)
            (video_shotname, extension) = os.path.splitext(tempfilename)
            folder_name = videodir + '_' + video_shotname

            # get the last two digits of the video_shotname using string slicing
            # last_two_digits = video_shotname[-2:]
            # alternatively, you can use regular expressions to get the last two digits
            # last_two_digits = re.search(r"\d{2}$", video_shotname).group()
            # construct the label as foldername_last2digitsofvideofile
            # label = folder_name + "_" + last_two_digits

            Path = os.path.join(Frame, folder_name)
            # print(Path)
            if not os.path.exists(Path):
                os.mkdir(Path)



            # frames = int(clip.fps * clip.duration)

            cap = cv2.VideoCapture(v)
            fps = cap.get(5)
            totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(fps,totalFrameNumber)
            i = 0
            while (cap.isOpened()):  # cv2.VideoCapture.isOpened()
                i = i + 1
                ret, frame = cap.read()  # cv2.VideoCapture.read()　
                if ret == True:
                    path = Path + '/'
                    picturepath = path + str('%02d' % i) + '.jpg'
                    # print(picturepath)
                    cv2.imwrite(picturepath, frame)

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
    #
