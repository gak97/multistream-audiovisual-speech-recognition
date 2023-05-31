
import cv2
import numpy as np
import scipy.ndimage as ndi
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import sys
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import io,measure,color,data,filters
import ROI
# from xlwt import *
# import xlwt
import csv
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import os
import glob

def Features(mouth_centroid_x, mouth_centroid_y,b,shotname,Gabor_path,SheetPath,FeaturesPath):
 #curdir = 'D:/codepython3/Sheet/' # path to store sheets
 if not os.path.exists(SheetPath):
        os.mkdir(os.path.join(SheetPath))
 # else:
 #        print 'file exist'
 cur_dir = os.path.join(SheetPath, shotname)
 if not os.path.exists(cur_dir):
     os.mkdir(cur_dir)

 #curdir2 = 'D:/codepython3/Features/'  # path to store sheets
 if not os.path.exists(FeaturesPath):
     os.mkdir(os.path.join(FeaturesPath))
 # else:
 #        print 'file exist'
 cur_dir2 = os.path.join(FeaturesPath, shotname)
 if not os.path.exists(cur_dir2):
     os.mkdir(cur_dir2)

 image =cv2.imread( Gabor_path,0)
 thresh = filters.threshold_yen(image)  # high thresh
 # thresh = filters.threshold_otsu(image)  #Determination of optimal Segmentation threshold by otsu algorithm
 bwimg = (image >= (thresh))  # Segmenting with threshold to generate binary image
 labels, num = measure.label(bwimg, return_num=True, background=1)  # Labeled connected region
 image_label_overlay = label2rgb(labels, image=image, bg_label=-1)

 #show picture2
 fig, ax = plt.subplots(figsize=(10, 6))
 ax.imshow(image_label_overlay)


 if num > 0:
     x1 = mouth_centroid_x
     y1 = mouth_centroid_y
     minw = 1000
     minh = 1000
     for region in measure.regionprops(labels, intensity_image=image):

         minr, minc, maxr, maxc = region.bbox
         area = region.area
         meanintensity = region.mean_intensity
         orientation = region.orientation
         x = region.centroid[1]
         y = region.centroid[0]


         w = abs(x1 - x)
         h = abs(y - y1)
             # print 'w,h',w,h

         # wah = w + h
         # print 'w,h',w,h


         # if wah < minwah:
         if w <= minw and h <= minh:
             minw = w
             minh = h
             # minwah = wah
             # minw = w
             # # print 'minw',minw
             # minh = h
             # print 'minh',minh
             min_maxc = maxc
             min_maxr = maxr
             min_minc = minc
             min_minr = minr
             min_area = area
             min_meanintensity = meanintensity
             min_orientation = orientation
             min_centroidx = x
             min_centroidy = y

     min_x = min_centroidx  # centroid
     min_y = min_centroidy
     # print 'minx,miny',minx,miny
     #show picture
     #rect = cv2.rectangle(image_label_overlay, (min_minc, min_minr), (min_maxc, min_maxr), (0, 0, 255), 1)
     rect = mpatches.Rectangle((min_minc, min_minr), min_maxc - min_minc, min_maxr - min_minr,
                              fill=False, edgecolor='red', linewidth=1)
     cir1 = mpatches.Circle((min_x, min_y), radius=0.5, color='y')
     #cir1 =cv2.circle(rect,(min_x, min_y),radius=1, color='y')
     ax.add_patch(cir1)
     ax.add_patch(rect)
     cur_dir2 = cur_dir2 +'/'   # 保存路径
     # print(path)
     if not os.path.exists(cur_dir2):
         os.mkdir(cur_dir2)
     Features_path = cur_dir2 + str('%02d' % b) + '.png'
     #cv2.imwrite(Features_path,image_label_overlay)

     plt.savefig(Features_path)
     plt.close('all')
     # write the weight, height, area ,mass to txt
     width = min_maxc - min_minc
     height = min_maxr - min_minr
     Final_area = min_area
     Final_meanintensity = min_meanintensity
     Final_orientation = min_orientation
     Final_centroidx = min_x
     Final_centroidy = min_y

     # print 'Final_area',Final_area
     # print 'Final_intensity',Final_meanintensity
     # print 'Final_oritentation', Final_orientation
     # print 'Final_centroidx',Final_centroidx
     # print 'Final_centroidy',Final_centroidy
     #show picture
     # ax.set_axis_off()
     # plt.tight_layout()
     # plt.show()

     # file = csv.Workbook()
     # table = file.add_sheet('Sheet1')
     # table.title = "Sheet1"
     #
     # filesheet = workbook.create_sheet()
     # filesheet.title = "new table"
     parameters=['Box_width', 'Box_height', 'Final_area', 'Centroid_x', 'Centroid_y', 'intensity', 'orientation']

     value=[width, height, Final_area,Final_centroidx,Final_centroidy, Final_meanintensity * Final_area, Final_orientation]

     # for i in range(0, len(parameters)):
     #     table.write(i ,0, str(parameters[i]))
     #
     # # 填入第二列
     # for i in range(0, len(value)):
     #     table.write(i , 1, float(value[i]))
     cur_dir = cur_dir + '/'   # 保存路径
     # print(path)
     if not os.path.exists(cur_dir):
         os.mkdir(cur_dir)
     sheetPath=cur_dir + str('%02d' % b) + '.csv'
     with open(sheetPath, 'w' ,newline='') as f:
         writer=csv.writer(f)
         for i in range(0, len(parameters)):
            writer.writerow([parameters[i],value[i]])




 if num == 0:
     width = 0
     height = 0
     Final_area = 0
     Final_meanintensity = 0
     Final_orientation = 0
     Final_centroidx = mouth_centroid_x
     Final_centroidy = mouth_centroid_y


     parameters = ['Box_width', 'Box_height', 'Final_area', 'Centroid_x', 'Centroid_y', 'intensity', 'orientation']

     value = [width, height, Final_area, Final_centroidx, Final_centroidy, Final_meanintensity * Final_area,
              Final_orientation]


     cur_dir = cur_dir +'/'   # 保存路径
     # print(path)
     if not os.path.exists(cur_dir):
         os.mkdir(cur_dir)
     sheetPath = cur_dir + str('%02d' % b) + '.csv'
     with open(sheetPath, 'w',newline='') as f:
         writer = csv.writer(f)
         for i in range(0, len(parameters)):
            writer.writerow([parameters[i],value[i]])
