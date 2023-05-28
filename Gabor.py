
import cv2
import numpy as np
import pylab as pl
from PIL import Image
from PIL import ImageFilter
import os
import glob
import Features

#     The input parameters are: 
#     HGamma: gamma value of Gabor filter
#     HKernelSize: Kernel size of Gabor filter
#     HSig: Sigma value of Gabor filter
#     HWavelength: Wavelength value of Gabor filter
#     i: the number of picture
#     ROIpath: the path of the picture
#     shotname: the name of the picture
#     GaborPath: the path to store Gabor features

def Gabor_h(HGamma, HKernelSize, HSig, HWavelength,i,ROIpath,shotname,GaborPath):

    #cur_dir2 = 'D:/codepython3/Gabor/' # path to store Gabor features

    if not os.path.exists(GaborPath):
        os.mkdir(os.path.join(GaborPath))
    Gaborpath = os.path.join(GaborPath, shotname)
    print(Gaborpath)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)

    img = cv2.imread(ROIpath, 1)  # Loading color picture
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
    imgGray_f = np.array(imgGray, dtype=np.float32)  # Change data type of picture
    imgGray_f /= 255.

    # Parameters of horizontal filter
    orientationH = 90  # orientation of normal direction
    wavelengthH = HWavelength
    kernel_sizeH = HKernelSize
    sigH = HSig
    gmH = HGamma

    # 1. orientationH: orientation of the normal to the parallel stripes of a Gabor function.
    # 2. kernel_sizeH: size of the Gabor filter
    # 3. sigH: standard deviation of the Gaussian function used to modulate the Gabor function.
    # 4. wavelengthH: wavelength of the sinusoidal factor in the above equation.
    # 5. gmH: spatial aspect ratio.
    # 6. ps: phase offset.

    ps = 0.0
    thH = orientationH * np.pi / 180
    kernelH = cv2.getGaborKernel((kernel_sizeH, kernel_sizeH), sigH, thH, wavelengthH, gmH, ps)
    destH = cv2.filter2D(imgGray_f, cv2.CV_32F, kernelH)  # CV_32F
    Gaborpath = Gaborpath + '/'   # 保存路径
    # print(path)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)
    Gabor_Path = Gaborpath  + str('%02d' % i) + '.jpg'
    cv2.imwrite(Gabor_Path, np.power(destH, 2))
    return Gabor_Path

    #return i,shotname,Gabor_Path