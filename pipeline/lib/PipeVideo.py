'''
   Pipeline Video Functions
'''

import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image, ImageChops
import datetime
import os
import glob

from lib.PipeImage import stack_frames_fast , stack_stack, mask_frame
from lib.PipeUtil import cfe, save_json_file, convert_filename_to_date_cam, get_masks
from lib.DEFAULTS import * 

def find_hd_file(sd_file, sd_start_trim, sd_end_trim, trim_on =1):
   print("SD/HD: ", sd_file)
   print("SD Trim Num: ", sd_start_trim)
   print("SD Trim Num End: ", sd_end_trim)

   dur_frames = sd_end_trim - sd_start_trim 

   (sd_datetime, sd_cam, sd_date, sd_y, sd_m, sd_d, sd_h, sd_M, sd_s) = convert_filename_to_date_cam(sd_file)
   #if sd_start_trim > 1400:
   #   hd_file, hd_trim = eof_processing(sd_file, trim_num, dur)
   #   time_diff_sec = int(trim_num / 25)
   #   if hd_file != 0:
   #      return(hd_file, hd_trim, time_diff_sec, dur)
   offset = int(sd_start_trim) / 25
   meteor_datetime = sd_datetime + datetime.timedelta(seconds=offset)
   hd_glob = "/mnt/ams2/HD/" + sd_y + "_" + sd_m + "_" + sd_d + "_*" + sd_cam + "*.mp4"
   hd_files = sorted(glob.glob(hd_glob))
   for hd_file in hd_files:
      el = hd_file.split("_")
      if len(el) == 8 and "meteor" not in hd_file and "crop" not in hd_file and "trim" not in hd_file:
         hd_datetime, hd_cam, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(hd_file)
         time_diff = meteor_datetime - hd_datetime
         time_diff_sec = time_diff.total_seconds()
         if 0 < time_diff_sec < 60:
            time_diff_sec = time_diff_sec 
            if sd_start_trim == 0:
               sd_start_trim = 1
            if time_diff_sec < 0:
               time_diff_sec = 0
            if trim_on == 1:
               print("TRIMTRIMTIRM")
               print("HD FILE:", hd_file, time_diff_sec)
               hd_trim_start = int(time_diff_sec * 25)
               hd_trim_end = int(time_diff_sec * 25) + int(dur_frames)
               hd_out = hd_file.replace(".mp4", "-trim-" + str(hd_trim_start) + "-HD.mp4")
               if cfe(hd_out) == 0:
                  ffmpeg_splice(hd_file, hd_trim_start, hd_trim_end , hd_out)
            else:
               print("NOOOOOOOOOOOOOOOOOOOOOO TRIMMMMMMMMMMMMMMM")
               hd_trim = None
            return(hd_file, hd_out, time_diff_sec )
   # No HD file was found. Trim out the SD Clip and then upscale it.
   print("NO HD FOUND!")

   time_diff_sec = int(trim_num / 25)
   dur = int(dur) + 1 + 3
   print("UPSCALE FROM SD!", time_diff_sec, dur)
   time_diff_sec = time_diff_sec - 1
   if "passed" in sd_file:
      sd_trim = ffmpeg_trim(sd_file, str(time_diff_sec), str(dur), "-trim" + str(o_trim_num) + "")
   else:
      sd_trim = ffmpeg_trim(sd_file, str(time_diff_sec), str(dur), "-trim-" + str(trim_num) + "-SD-meteor")
   hd_trim = upscale_sd_to_hd(sd_trim)
   if "-SD-meteor-HD-meteor" in hd_trim:
      orig_hd_trim = hd_trim
      hd_trim = hd_trim.replace("-SD-meteor", "")
      hdf = hd_trim.split("/")[-1]
      os.system("mv " + orig_hd_trim + " /mnt/ams2/HD/" + hdf)
      print("HD F: mv " + orig_hd_trim + " /mnt/ams2/HD/" + hdf)
      hd_trim = "/mnt/ams2/HD/" + hdf

   return(sd_file,hd_trim,str(0),str(dur))


def ffmpeg_splice(video_file, start, end, outfile):

   cmd = "/usr/bin/ffmpeg -i " + video_file + " -vf select='between(n\," + str(start) + "\," + str(end) + ")' -vsync 0 -start_number " + str(start) + " " + outfile + " > /dev/null 2>&1 "


   print(cmd)
   os.system(cmd)


def scan_stack_file(file, vals = []):

   start_time = time.time()

   fn = file.split("/")[-1]
   day = fn[0:10]
   proc_dir = PROC_BASE_DIR + "/" + day + "/"
   proc_img_dir = proc_dir + "images/"
   proc_data_dir = proc_dir + "data/"
   if cfe(proc_img_dir, 1) == 0:
      os.makedirs(proc_img_dir)
   if cfe(proc_data_dir, 1) == 0:
      os.makedirs(proc_data_dir)
   stack_file = proc_img_dir + fn.replace(".mp4", "-stacked-tn.png")
   json_file = proc_data_dir + fn.replace(".mp4", "-vals.json")

   frames = []
   gray_frames = []
   sub_frames = []

   sum_vals = []
   max_vals = []
   avg_max_vals = []
   pos_vals = []
   fd = []

   stacked_image = None
   fc = 0

   cap = cv2.VideoCapture(file)

   while True:
      grabbed , frame = cap.read()
      #if fc < len(vals):
      #   if vals[fc] == 0  and fc > 20:
      #      print("SKIP FRAME:", fc, vals[fc])
      #      fc = fc + 1
      #      continue

      if not grabbed and fc > 5:
         print(fc)
         break

      try:
         small_frame = cv2.resize(frame, (0,0),fx=.5, fy=.5)
      except:
         print("Bad video file:", file)


      if True:
         gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
         if fc > 0:
            sub = cv2.subtract(gray, gray_frames[-1])
         else:
            sub = cv2.subtract(gray, gray)

         min_val, max_val, min_loc, (mx,my)= cv2.minMaxLoc(sub)
         if max_val < 10:
            sum_vals.append(0)
            max_vals.append(0)
            pos_vals.append((0,0))
         else:
            _, thresh_frame = cv2.threshold(sub, 15, 255, cv2.THRESH_BINARY)
            #min_val, max_val, min_loc, (mx,my)= cv2.minMaxLoc(thresh_frame)
            sum_val =cv2.sumElems(thresh_frame)[0]
            #mx = mx * 2
            #my = my * 2
            sum_vals.append(sum_val)
            max_vals.append(max_val)
            if max_val > 1:
               avg_max_vals.append(max_val)
            pos_vals.append((mx,my))
         gray_frames.append(gray)

      if True:
         if max_val > 10 or fc < 10:
            avg_max = np.median(avg_max_vals)
            if avg_max > 0:
               diff = (max_val / avg_max) * 100
            else:
               diff = 0
            if max_val > avg_max * 1.2 or fc <= 10:
               #print("STAK THE FRAME", avg_max, max_val, diff, fc)
               frame_pil = Image.fromarray(small_frame)
               if stacked_image is None:
                  stacked_image = stack_stack(frame_pil, frame_pil)
               else:
                  stacked_image = stack_stack(stacked_image, frame_pil)

      frames.append(frame)
      if fc % 100 == 1:
         print(fc)
      fc += 1
   cv_stacked_image = np.asarray(stacked_image)
   cv_stacked_image = cv2.resize(cv_stacked_image, (PREVIEW_W, PREVIEW_H))
   cv2.imwrite(stack_file, cv_stacked_image)
   print(stack_file)


   vals = {}
   vals['sum_vals'] = sum_vals
   vals['max_vals'] = max_vals
   vals['pos_vals'] = pos_vals
   if cfe(stack_file) == 0:
      #logger("scan_stack.py", "scan_and_stack_fast", "Image file not made! " + stack_file + " " )
      print("ERROR: Image file not made! " + stack_file)
      #time.sleep(10)
   save_json_file(json_file, vals)
   elapsed_time = time.time() - start_time
   #os.system("mv " + file + " " + proc_dir)
   print("saved.", json_file)

   if cfe(stack_file) == 0:
      print("No stack file made!?")
      logger("scan_stack.py", "scan_and_stack_fast", "Image file not made! " + stack_file + " " )
      exit()

   # mv video file if it is not already in proc2 dir
   if "proc2" not in file:
      cmd = "mv " + file + " " + proc_dir
      print(cmd)
   else:
      print("File already in proc dir!")

   print("Elp:", elapsed_time)
   


def load_frames_fast(trim_file, json_conf, limit=0, mask=0,crop=(),color=0,resize=[], sun_status="night"):
   (f_datetime, cam, f_date_str,fy,fm,fd, fh, fmin, fs) = convert_filename_to_date_cam(trim_file)
   cap = cv2.VideoCapture(trim_file)

   if "HD" in trim_file:
      masks = get_masks(cam, json_conf,1)
   else:
      masks = get_masks(cam, json_conf,1)
   if "crop" in trim_file:
      masks = None

   color_frames = []
   frames = []
   subframes = []
   sum_vals = []
   pos_vals = []
   max_vals = []
   frame_count = 0
   last_frame = None
   go = 1
   while go == 1:
      if True :
         _ , frame = cap.read()
         if frame is None:
            if frame_count <= 5 :
               cap.release()
               return(frames,color_frames,subframes,sum_vals,max_vals,pos_vals)
            else:
               go = 0
         else:
            if color == 1:
               if sun_status == "day" and frame_count % 25 == 0:
                  color_frames.append(frame)
               else:
                  color_frames.append(frame)
            if limit != 0 and frame_count > limit:
               cap.release()
               return(frames,color_frames,subframes,sum_vals,max_vals,pos_vals)
            if len(resize) == 2:
               frame = cv2.resize(frame, (resize[0],resize[1]))

            if True: 
               frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               if mask == 1 and frame is not None:
                  if frame.shape[0] == 1080:
                     hd = 1
                  else:
                     hd = 0
                  masks = get_masks(cam, json_conf,hd)
                  frame = mask_frame(frame, [], masks, 5)

               if last_frame is not None:
                  subframe = cv2.subtract(frame, last_frame)
                  sum_val =cv2.sumElems(subframe)[0]

                  if sum_val > 10 :
                     _, thresh_frame = cv2.threshold(subframe, 15, 255, cv2.THRESH_BINARY)

                     sum_val =cv2.sumElems(thresh_frame)[0]
                  else: 
                     sum_val = 0
                  subframes.append(subframe)


                  if sum_val > 10:
                     min_val, max_val, min_loc, (mx,my)= cv2.minMaxLoc(subframe)
                  else:
                     max_val = 0
                     mx = 0
                     my = 0
                  if frame_count < 5:
                     sum_val = 0
                     max_val = 0
                  sum_vals.append(sum_val)
                  max_vals.append(max_val)
                  pos_vals.append((mx,my))
               else:
                  blank_image = np.zeros((frame.shape[0] ,frame.shape[1]),dtype=np.uint8)
                  subframes.append(blank_image)
                  sum_val = 0
                  sum_vals.append(0)
                  max_vals.append(0)
                  pos_vals.append(0)

            frames.append(frame)
            last_frame = frame
      frame_count = frame_count + 1
   cap.release()
   return(frames, color_frames, subframes, sum_vals, max_vals,pos_vals)

#def get_masks():


#def find_hd_file():


#def trim_crop_video():
