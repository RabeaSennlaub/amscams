import os
import glob
import subprocess  
import json
from lib.Frame_Tools import *  

meteor_json_file = '/mnt/ams2/meteors/2019_07_31/2019_07_31_09_22_50_000_010040-trim1090.json'
sd_video_file = '/mnt/ams2/meteors/2019_07_31/2019_07_31_09_22_50_000_010040-trim1090.mp4'

#def add_frame(json_conf, sd_video_file, fr_id, hd_x=-1, hd_y=-1): 
#add_frame(meteor_json_file,sd_video_file,str(22),251,123) 
real_add_frame(meteor_json_file,sd_video_file,str(44),407,247) 
 