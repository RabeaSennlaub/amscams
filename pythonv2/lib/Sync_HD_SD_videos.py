import cv2
import sys
import os  
import numpy as np
from lib.FileIO import load_json_file, save_json_file, cfe
from lib.VIDEO_VARS import *
from lib.UtilLib import convert_filename_to_date_cam
from lib.Get_Cam_ids import get_the_cameras


# Define mask frame
def mask_frame(frame, mp, masks, size=3):
  
   hdm_x = HD_W/SD_W
   hdm_y = HD_H/SD_H

   # Mask bright pixels detected in the median  and also mask areas defined in the config
   frame.setflags(write=1)
   ih,iw = frame.shape[0], frame.shape[1]
   px_val = np.mean(frame)
   px_val = 0

   for mask in masks:
      mx,my,mw,mh = mask.split(",")
      mx,my,mw,mh = int(mx), int(my), int(mw), int(mh) 
      frame[int(my):int(my)+int(mh),int(mx):int(mx)+int(mw)] = 0

   for x,y in mp:

      if int(y + size) > ih:
         y2 = int(ih - 1)
      else:
         y2 = int(y + size)
      if int(x + size) > iw:
         x2 = int(iw - 1)
      else:
         x2 = int(x + size)

      if y - size < 0:
         y1 = 0
      else:
         y1 = int(y - size)
      if int(x - size) < 0:
         x1 = 0
      else:
         x1 = int(x - size)

      x1 = int(x1)
      x2 = int(x2)
      y1 = int(y1)
      y2 = int(y2)

      frame[y1:y2,x1:x2] = px_val
   return(frame)

# Get frame mask
def get_masks(this_cams_id, hd = 0):  

   my_masks = []
   cameras = get_the_cameras()
  
   for camera in cameras:
      if str(cameras[camera]['cams_id']) == str(this_cams_id):
         if hd == 1:
            masks = cameras[camera]['hd_masks']
         else:
            masks = cameras[camera]['masks']
         for key in masks:
            mask_el = masks[key].split(',')
            (mx, my, mw, mh) = mask_el
            masks[key] = str(mx) + "," + str(my) + "," + str(mw) + "," + str(mh)
            my_masks.append((masks[key]))
   return(my_masks)

# Return video frames
def load_video_frames(trim_file, json_conf, limit=0, mask=0, color=0):

   (f_datetime, cam, f_date_str,fy,fm,fd, fh, fmin, fs) = convert_filename_to_date_cam(trim_file)
 
   cap = cv2.VideoCapture(trim_file)
   masks = None 
   frames = []
   frame_count = 0
   go = 1
   
   while go == 1:
      _ , frame = cap.read()
      #print(frame_count)
      if frame is None:
         if frame_count <= 5 :
            cap.release()
            return(frames)
         else:
            go = 0
      else:
         if limit != 0 and frame_count > limit:
            cap.release()
            return(frames)
         if len(frame.shape) == 3 and color == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

         if mask == 1 and frame is not None:
            if frame.shape[0] == HD_H:
               hd = 1
            else:
               hd = 0
            masks = get_masks(cam, hd)
            #print("GET MASKS HD:", hd, masks)
            frame = mask_frame(frame, [], masks, 5)

         frames.append(frame)
         frame_count = frame_count + 1
   cap.release()
   
   return(frames)


# Try so sync HD & SD video
def sync_hd_frames(hd_video_file,sd_video_file,json_reduction_file):
   
   print("IN sync_hd_frames")

   reduction_data = load_json_file(json_reduction_file)

   print("reduction_data")
   print(reduction_data)

   # Get the HD Frames
   hd_frames = load_video_frames(hd_video_file, json_reduction_file, limit=0, mask=1, color=1)
   
   # Get the SD Frames
   sd_frames = load_video_frames(sd_video_file, json_reduction_file,  limit=0, mask=1, color=1)
 

   metframes = reduction_data['metframes']
   first_sd_frame = None
   first_hd_frame = None
   hd_fns = []
   sd_fns = []

   print("METFRAMES")
   print(metframes)

   for fn in metframes:
      if first_sd_frame is None:
         first_sd_fram = fn
      x1 = metframes[fn]['x1']
      x2 = metframes[fn]['x2']
      y1 = metframes[fn]['y1']
      y2 = metframes[fn]['y2']
      hd_x = metframes[fn]['hd_x']
      hd_y = metframes[fn]['hd_y']
      hd_fn = find_hd_frame(fn, hd_x, hd_y, x1,y1,x2,y2,hd_frames)
      sd_fns.append(int(fn))
      hd_fns.append(int(hd_fn))
      print(fn, metframes[fn]['hd_x'], metframes[fn]['hd_y'])
   
   print("len(sd_fns): " + len(sd_fns))   
   print("len(hd_fns): " + len(hd_fns))   

   if len(sd_fns) == len(hd_fns):
      # buffer the frames with 10 frames on either side if we can.
      hd_archive_movie = sd_video_file.replace(".mp4", "-archiveHD.mp4")
      sd_archive_movie = sd_video_file.replace(".mp4", "-archiveSD.mp4")
      hd_start_buff, hd_end_buff = make_movie_from_frames(hd_frames, hd_fns, hd_archive_movie)
      sd_start_buff, sd_end_buff = make_movie_from_frames(sd_frames, sd_fns, sd_archive_movie)
      reduction_data['metconf']['archive_sd_pre_roll'] = sd_start_buff
      reduction_data['metconf']['archive_sd_post_roll'] = sd_end_buff
      reduction_data['metconf']['archive_hd_pre_roll'] = hd_start_buff
      reduction_data['metconf']['archive_hd_post_roll'] = hd_end_buff
      reduction_data['metconf']['hd_sync'] = 1 
      print("Perfect HD/SD frame match up!")       
      print("SD FRAMES:", sd_fns)
      print("HD FRAMES:", hd_fns)
      print("Archive HD Movie:", hd_archive_movie)
      print("Archive SD Movie:", sd_archive_movie)
      save_json_file(json_reduction_file, reduction_data)


def find_hd_frame(fn, hd_x, hd_y, x1,y1,x2,y2,hd_frames):
   crops = []
   max_hd_val = 0
   best_hd_frame = 0
   cc = 0
   best_cc = 0
   for frame in hd_frames:
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      crop_frame = gray_frame[y1:y2,x1:x2]
      max_val = gray_frame[hd_y,hd_x]
      if max_val > max_hd_val:
         max_hd_val = max_val
         best_hd_frame = cc 
         best_cc = cc
      cc = cc + 1
      crops.append(crop_frame)
   #cv2.imshow('pepe', crops[best_cc])
   #cv2.waitKey(0)
   return(best_hd_frame)