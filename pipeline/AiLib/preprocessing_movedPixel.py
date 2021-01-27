# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:29:37 2021

@author: Voovo
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import json
from pathlib import Path
import os
import deepdish as dd

def preprocessing(pathVid):
    # check if valid input file
    if pathVid.endswith(".mp4"):
        # cut out images around brightest frame
        img_stack = []
        # meta data of img_stack (frame_nr, y, x)
        meta_stack = []

        # array of frames in video
        frames = []

        vidcap = cv2.VideoCapture(pathVid)
        success,frame = vidcap.read()

        # get all the frames from crop vid
        while success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            success,frame = vidcap.read()

        coords = []

        cnt = 0
        frame_nr = 0

        old_indx = (0,0)

        # get coords of brightest pixel in each frame
        for img in frames:
            img_pad = np.pad(img,16)
            # max pixel and coord
            min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(img_pad)

            # check if brightest Pixel has moved
            if (frame_nr == 0 or (abs(np.array(old_indx)-np.array(max_indx))>0).any()) and (abs(np.array(old_indx)-np.array(max_indx))<30).all():

                # treshold for minimal brightness
                if max_val > 50:
                    coords.append(max_indx)
                    pathSquare = pathVid[:-4]+str(cnt)+".jpg"
                    cv2.imwrite(pathSquare, img_pad[max_indx[1]-16:max_indx[1]+16,max_indx[0]-16:max_indx[0]+16])
                    img_crop = img_pad[max_indx[1]-16:max_indx[1]+16,max_indx[0]-16:max_indx[0]+16]
                    img_stack.append(img_crop)
                    meta_stack.append(np.array([frame_nr, max_indx[1], max_indx[0], max_val]))
                    cnt = cnt+1
            frame_nr = frame_nr+1
            old_indx = max_indx

        #lineImg = cv2.imread(pathImg,cv2.IMREAD_COLOR)

        vid_dict[pathVid[-43:]] = [np.array(img_stack), np.array(meta_stack)]

        # draw coords of brightest pixel in image
        #for point in coords:
        #    cv2.circle(lineImg, point, radius=1, color=(0, 0, 255), thickness=-1)

        # save image with line in it
        #cv2.imwrite(pathOutLine, lineImg)


# define dict
# key: vidPath
# value: [img_stack, meta_stack] with img_stack np array (#frames,32,32) and meta_stack np array (#frames,3)
vid_dict = dict([])


vidFolder = Path(r'D:\Voovo\Documents\Uni\Masterarbeit\Data_Sirko\detections_ams35\matched\Zwischenstaende\\')

prog = 0
for x in vidFolder.iterdir():
    if str(x).endswith("_crop.mp4"):
        preprocessing(str(x))
        print(prog)
        prog +=1

#dd.io.save(r'D:\Voovo\Documents\Uni\Masterarbeit\Data_Sirko\detections_ams35\matched\meteor\meteor_hd.h5', vid_dict)