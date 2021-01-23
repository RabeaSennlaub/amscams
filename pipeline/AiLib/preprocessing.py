# -*- coding: utf-8 -*-
import numpy as np
import cv2
import json

def preprocessing(pathVid, pathImg, pathOutLine):
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

        # get coords of brightest pixel in each frame
        for img in frames:
                # max pixel and coord
                min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(img)
                # treshold for minimal brightness
                if max_val > 50:
                    coords.append(max_indx)
                    pathSquare = pathImg[:-4]+str(cnt)+".jpg"
                    cv2.imwrite(pathSquare, img[max_indx[1]-16:max_indx[1]+16,max_indx[0]-16:max_indx[0]+16])
                    img_stack.append(img[max_indx[1]-16:max_indx[1]+16,max_indx[0]-16:max_indx[0]+16])
                    meta_stack.append(np.array([frame_nr, max_indx[1], max_indx[0]]))
                    cnt = cnt+1
                frame_nr = frame_nr+1

        lineImg = cv2.imread(pathImg,cv2.IMREAD_COLOR)

        vid_dict[pathVid] = [np.array(img_stack), np.array(meta_stack)]

        # draw coords of brightest pixel in image
        for point in coords:
            cv2.circle(lineImg, point, radius=1, color=(0, 0, 255), thickness=-1)

        # save image with line in it
        cv2.imwrite(pathOutLine, lineImg)


# define dict
# key: vidPath
# value: [img_stack, meta_stack] with img_stack np array (#frames,32,32) and meta_stack np array (#frames,3)
vid_dict = dict([])
preprocessing(r'D:\Voovo\Documents\Uni\Masterarbeit\Data_Sirko\detections_ams35\matched\Zwischenstaende\2020_03_26_20_38_32_000_010073-trim0906_crop.mp4',r'D:\Voovo\Documents\Uni\Masterarbeit\Data_Sirko\detections_ams35\matched\Zwischenstaende\2020_03_26_20_38_32_000_010073-trim0906_crop.jpg', r'D:\Voovo\Documents\Uni\Masterarbeit\Data_Sirko\detections_ams35\matched\Zwischenstaende\2020_03_26_20_38_32_000_010073-trim0906_crop_line.jpg')