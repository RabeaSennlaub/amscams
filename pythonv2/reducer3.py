#!/usr/bin/python3

import datetime
import time
import glob
import os
import math
import cv2
import math
import numpy as np
import scipy.optimize
from lib.VideoLib import get_masks, find_hd_file_new, load_video_frames
from lib.UtilLib import check_running, angularSeparation
from lib.CalibLib import radec_to_azel, clean_star_bg, get_catalog_stars, find_close_stars, XYtoRADec, HMS2deg, AzEltoRADec

from lib.ImageLib import mask_frame , stack_frames, preload_image_acc
from lib.ReducerLib import setup_metframes, detect_meteor , make_crop_images, perfect
from lib.MeteorTests import meteor_test_cm_gaps


#import matplotlib.pyplot as plt
import sys
#from caliblib import distort_xy,
from lib.CalibLib import distort_xy_new, find_image_stars, distort_xy_new, XYtoRADec, radec_to_azel, get_catalog_stars,AzEltoRADec , HMS2deg, get_active_cal_file, RAdeg2HMS, clean_star_bg
from lib.UtilLib import calc_dist, find_angle, bound_cnt, cnt_max_px

from lib.UtilLib import angularSeparation, convert_filename_to_date_cam, better_parse_file_date
from lib.FileIO import load_json_file, save_json_file, cfe
from lib.UtilLib import calc_dist,find_angle
import lib.brightstardata as bsd
from lib.DetectLib import eval_cnt, id_object
json_conf = load_json_file("../conf/as6.json")

cmd = sys.argv[1]
file = sys.argv[2]
try:
   show = int(sys.argv[3])
except:
   show = 0

if cmd == 'dm' or cmd == 'detect_meteor':
   metframes, frames, metconf = detect_meteor(file, json_conf, show)
   print("Metframes")
   for fn in metframes:
      print(fn, metframes[fn])

if cmd == 'cm' or cmd == 'crop_images':
   make_crop_images(file, json_conf)

#MFD TO METFRAMES
if cmd == 'mfd' :
   # perfect the meteor reduction!
   if "mp4" in file:
      file = file.replace(".mp4", "-reduced.json")
   red_data = load_json_file(file)
   mfd = red_data['meteor_frame_data']
   metframes, metconf = setup_metframes(mfd)
   red_data['metframes'] = metframes 
   red_data['metconf'] = metconf 
   save_json_file(file, red_data)

if cmd == 'pf' or cmd == 'perfect':
   # perfect the meteor reduction!
   perfect(file, json_conf)


