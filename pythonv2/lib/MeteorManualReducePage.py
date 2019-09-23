import cgitb
import re
import os
import sys
import glob

from lib.FileIO import load_config
from lib.MeteorReducePage import print_error
from lib.MeteorReduce_Tools import *
from lib.REDUCE_VARS import *
from lib.VIDEO_VARS import *

MANUAL_RED_PAGE_TEMPLATE = "/home/ams/amscams/pythonv2/templates/manual_reduction_template.html"


# Fix the old files names that contains "-trim"
def fix_old_file_name(filename):
   # We need to get the current stations ID (in as6.json)
   json_conf = load_json_file(JSON_CONFIG)
   station_id = json_conf['site']['ams_id']
   if("-reduced" in filename):
      filename = filename.replace("-reduced", "")

   if("trim" in filename):
      tmp_video_full_path_matches =  re.finditer(OLD_FILE_NAME_REGEX, filename, re.MULTILINE)
      tmp_fixed_video_full_path = ""
      for matchNum, match in enumerate(tmp_video_full_path_matches, start=1):
         for groupNum in range(0, len(match.groups())): 
            if("-" not in match.group(groupNum)):
               tmp_fixed_video_full_path = tmp_fixed_video_full_path + "_" + match.group(groupNum)
            groupNum = groupNum + 1

         # Remove first "_"
         tmp_fixed_video_full_path = tmp_fixed_video_full_path[1:]
         # Add an extension
         tmp_fixed_video_full_path += "_" + station_id
         
         if("HD" in filename):
            tmp_fixed_video_full_path +=  "_HD.json"
         else:
            tmp_fixed_video_full_path +=  "_SD.json"
         return tmp_fixed_video_full_path
   else:
      return filename




# First Step of the Manual reduction: select start / end meteor position
def manual_reduction(form):
   
   # Debug
   cgitb.enable()

   video_file = form.getvalue('video_file')

   # Build the page based on template  
   with open(MANUAL_RED_PAGE_TEMPLATE, 'r') as file:
      template = file.read()

   # Here we have the possibility to "empty" the cache, ie regenerate the files (stacks) even if they already exists
   # we just need to add "clear_cache=1" to the URL
   if(form.getvalue("clear_cache") is not None):
      clear_cache = True
   else:
      clear_cache = False

   # Get Video File & Analyse the Name to get quick access to all info
   video_full_path = form.getvalue("video_file")

   if(video_full_path is not None):
      tmp_fixed_video_full_path = fix_old_file_name(video_full_path)
      analysed_name = name_analyser(tmp_fixed_video_full_path)

      # We keep the original full_path anyway
      analysed_name['full_path'] = video_full_path
   else:
      print_error("<b>You need to add a video file in the URL.</b>")

   # Get the related JSON
   json_file = video_full_path.replace('.mp4','.json')
   template = template.replace("{JSON_FILE}", str(json_file))   # JSON File  

   # Get the stacks 
   # True = We automatically resize the stack to HD dims so we can use it in the UI
   stack = get_stacks(analysed_name,clear_cache, True)
   template = template.replace("{STACK}", str(stack))   
   
   # Add Video to template
   template = template.replace("{VIDEO}", str(video_file))

   # Display Template
   print(template)


# Second Step of Manual Reduction: cropp of all frames + selection of start event
def manual_reduction_cropper(form):

   video_file  = form.getvalue('video_file') 
   x_start = float(form.getvalue('x_start'))
   y_start = float(form.getvalue('y_start'))
   w = float(form.getvalue('w'))
   h = float(form.getvalue('h'))

   # Fix eventual video file name (old version)
   tmp_fixed_video_full_path = fix_old_file_name(video_file)
   analysed_name = name_analyser(tmp_fixed_video_full_path)

   # We keep the original full_path anyway
   analysed_name['full_path'] = video_file

   # Create destination folder if necessary
   dest_folder = get_cache_path(analysed_name,'tmp_cropped')
   cache_path  = does_cache_exist(analysed_name,'tmp_cropped')

   # If we already tmp cropped frames, we need to delete them
   if(len(cache_path)!=0):
      for f in cache_path:
         os.remove(os.path.join(cache_path, f))

   # Extract all the frames, resize to HD and crop
   cmd = 'ffmpeg   -i ' + analysed_name['full_path'] +  ' -filter_complex "[0:v]scale=' + str(HD_W) + ":" + str(HD_H) + '[scale];[scale]crop='+str(w)+':'+str(h)+':'+str(x_start)+':'+str(y_start)+'[out]"  -map "[out]" ' + dest_folder + '/%04d' + '.png' 
   output = subprocess.check_output(cmd, shell=True).decode("utf-8")  
   
   # Get all the newly created cropped frames
   thumbs = sorted(glob.glob(dest_folder+'/*.png'))
   
   #for img in thumbs:
   #   print("<img src='"+img+'"/>)