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

        coords_diff = []
        coords_orig = []

        cnt = 0
        frame_nr = 0

        # array of diff-images of consecutive frames
        diffs = []

        # for each frame
        for i in range(1,len(frames)):
            # calculate diff of frame and its previous frame
            diffs.append(cv2.absdiff(frames[i-1],frames[i]))

        # get coords of brightest pixel in each frame
        for img in diffs:
            img_pad = np.pad(img,16)
            img_pad_orig = np.pad(frames[frame_nr],16)

            # max pixel and coord in diff
            min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(img_pad)

            # max pixel and coord in orig
            min_val_orig,max_val_orig,min_indx_orig,max_indx_orig=cv2.minMaxLoc(img_pad_orig)

            # treshold for minimal brightness
            if max_val > 50 and max_val_orig > 50:
                coords_diff.append(max_indx)
                coords_orig.append(max_indx_orig)
                pathSquare = pathVid[:-4]+str(cnt)+".jpg"
                cv2.imwrite(pathSquare, img_pad_orig[max_indx_orig[1]-16:max_indx_orig[1]+16,max_indx_orig[0]-16:max_indx_orig[0]+16])
                img_crop = img_pad_orig[max_indx_orig[1]-16:max_indx_orig[1]+16,max_indx_orig[0]-16:max_indx_orig[0]+16]
                img_stack.append(img_crop)
                meta_stack.append(np.array([frame_nr, max_indx_orig[1], max_indx_orig[0], max_val_orig]))
                cnt = cnt+1
            frame_nr = frame_nr+1

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