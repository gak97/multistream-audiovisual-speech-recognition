#-*- coding: utf-8 -*-
import cv2
import os
import TPE
import Gabor

#face
def rect1(detector,predictor,i,shotname,picturepath,MouthPath,GaborPath,SheetPath,FeaturesPath):


 img = cv2.imread(picturepath)
 dets = detector(img, 1)

 # create a file PATH to store mouth picture

 if not os.path.exists(MouthPath):
    os.mkdir(os.path.join(MouthPath))
 cur_dir =os.path.join(MouthPath, shotname)
 if not os.path.exists(cur_dir):
     os.mkdir(os.path.join(cur_dir))
 # else:
 #     print 'file exist'


    #face
 for k, d in enumerate(dets):
     shape = predictor(img, d)

     t1 = shape.part(48).x - 10
     t2 = shape.part(54).x + 10
     t3 = shape.part(50).y - 5
     t4 = shape.part(58).y + 5

     mouth_centroid_x = (shape.part(48).x - t1) + abs(shape.part(54).x - shape.part(48).x) / 2
     mouth_centroid_y = (shape.part(51).y - t3) + abs(shape.part(62).y - shape.part(51).y) + abs(
         shape.part(66).y - shape.part(62).y) / 2

     # print("ROI image=",t1,t2,t3,t4)
     ROI_mouth = img[t3:t4, t1:t2]

     widthG = shape.part(64).x - shape.part(60).x
     heightG = shape.part(66).y - shape.part(62).y
     cur_dir = cur_dir + '/'   # 保存路径
     # print(path)


     if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
     ROIpath = cur_dir  + str('%02d' % i) + '.jpg'

     cv2.imwrite(ROIpath, ROI_mouth)
     return ROIpath,mouth_centroid_x, mouth_centroid_y, ROI_mouth, widthG, heightG



