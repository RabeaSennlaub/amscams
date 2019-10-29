import json
import re
import os
import glob
import shutil
import sys
import cgitb
import random

from datetime import datetime,timedelta

from lib.WebCalib import get_active_cal_file
from lib.FileIO import load_json_file,save_json_file, cfe
from lib.REDUCE_VARS import *
from lib.Get_Station_Id import get_station_id
from lib.VIDEO_VARS import FPS_HD
from lib.Sync_HD_SD_videos import * 
from lib.CGI_Tools import redirect_to
from lib.MeteorReduce_Tools import  name_analyser, old_name_analyser


# Return the analysed  version of the file name
# no matter if it's an old or a new file
def get_analysed_name(video_file):
   # We need to get the info from the file name either if it's an old file or a new file (in the archive)
   if(METEOR_ARCHIVE in video_file):
      analysed_name = name_analyser(video_file)
   else:
      analysed_name = old_name_analyser(video_file)
   
   # We keep the original full_path anyway
   analysed_name['full_path'] = video_file  

   # Do we have the station ID?
   if('station_id' not in analysed_name):
      analysed_name['station_id'] = get_station_id()
   
   return analysed_name

# Get a new folder in meteor_archive
# from an old json file
def get_new_archive_folder(analysed_name):
   if('station_id' not in analysed_name):
      station_id = get_station_id()
   else:
      station_id = analysed_name['station_id'] 
   return METEOR_ARCHIVE + station_id + "/" + METEOR + analysed_name['year'] + "/" + analysed_name['month'] + "/" + analysed_name['day'] + "/"

# Analysed an old file (containing "-trim")
# Parses a regexp (OLD_FILE_NAME_REGEX) a file name
# and returns all the info defined in OLD_FILE_NAME_REGEXGROUP
def old_name_analyser(file_names):
   matches = re.finditer(OLD_FILE_NAME_REGEX, file_names, re.MULTILINE)
   res = {}
  
   for matchNum, match in enumerate(matches, start=1):
      for groupNum in range(0, len(match.groups())):
         if(match.group(groupNum) is not None):
            res[OLD_FILE_NAME_REGEX_GROUP[groupNum]] = match.group(groupNum)
         groupNum = groupNum + 1

   # Get Name without extension if possible
   if(res is not None and "name" in res):
      res['name_w_ext'] = res['name'].split('.')[0]

   # Add the full file_names (often a full path) to the array so we don't have to pass the original when we need it
   res['full_path'] = file_names

   return res


# Fix the old files names that contains "-trim"
# so we can use the usual name_analyser
def fix_old_file_name(filename):

   # We need to get the current stations ID (in as6.json)
   json_conf = load_json_file(JSON_CONFIG)
   station_id = json_conf['site']['ams_id']
   if("-reduced" in filename):
      filename = filename.replace("-reduced", "")

   if("-stacked-calparams" in filename):
      filename = filename.replace("-stacked-calparams", "")
 
 
   trim_value = 0 

   if("trim" in filename):

      matches =  re.finditer(OLD_FILE_NAME_REGEX, filename, re.MULTILINE)
  
      res = {}
 
      for matchNum, match in enumerate(matches, start=1):
         for groupNum in range(0, len(match.groups())):
            if(match.group(groupNum) is not None): 
               res[OLD_FILE_NAME_REGEX_GROUP[groupNum]] = match.group(groupNum)
            groupNum = groupNum + 1
 
      # Get original Date & Time 
      org_dt = datetime.strptime(res['year']+'-'+res['month']+'-'+res['day']+' '+res['hour']+':'+res['min']+':'+res['sec']+'.'+res['ms'], '%Y-%m-%d %H:%M:%S.%f')

      # We convert the trim in seconds 
      if(res["trim"] is not None):
         trim_in_sec = float(res["trim"])/FPS_HD
      else:
         trim_in_sec = 0

      # We add the trim_in_sec
      org_dt = org_dt +  timedelta(0,trim_in_sec)

      # Create fixed name based on all data
      org_dt = org_dt.strftime("%Y_%m_%d_%H_%M_%S_%f")

      # [:-3] to only keep 4 digits for the microseconds
      toReturn =   org_dt[:-3] + '_'+res['cam_id']+'_'+get_station_id()

      if("HD" in filename):
         toReturn +=  "_HD.json"
      else:
         toReturn +=  "_SD.json"
 
      return toReturn
   else:
      return filename



# Get cal_params new version from an old JSON version 
def get_new_calib(json_f):

   # If 'device_alt' isn't defined, we have to work with 'site_alt'...
   if "device_alt" not in json_f['cal_params']:
      json_f['cal_params']['device_alt'] = float(json_f['cal_params']['site_alt'])
      json_f['cal_params']['device_lat'] = float(json_f['cal_params']['site_lat'])  
      json_f['cal_params']['device_lng'] = float(json_f['cal_params']['site_lng'])  
    
   new_dt = json_f['event_start_time']
   new_dt = new_dt.replace('/','_')
   new_dt = new_dt.replace(' ','_')
   new_dt = new_dt.replace(':','_')
   new_dt = new_dt.replace('.','_')
   new_dt = new_dt.replace('-','_')
  
   return { "calib":  
      {  "dt":   new_dt,
         "device": {
            "alt":  float(json_f['cal_params']['device_alt']),
            "lat":  float(json_f['cal_params']['device_lat']),
            "lng":  float(json_f['cal_params']['device_lng']),
            "scale_px":  float(json_f['cal_params']['pixscale']),
            "poly": {
                  "y_fwd": json_f['cal_params']['y_poly_fwd'],
                  "x_fwd": json_f['cal_params']['x_poly_fwd'],
                  "y": json_f['cal_params']['y_poly'],
                  "x": json_f['cal_params']['x_poly']
            },
            "center": {
                  "az": float(json_f['cal_params']['center_az']),  
                  "ra": float(json_f['cal_params']['ra_center']), 
                  "el": float(json_f['cal_params']['center_el']),
                  "dec": float(json_f['cal_params']['dec_center']) 
            },
            "angle":  float(json_f['cal_params']['position_angle']),
      }      
   }}

# Get new info (device & detection info) from an old JSON version
def get_new_info(json_f): 
  return  {
      "info": {
         "station": json_f['station_id'],
         "hd_vid":  json_f['hd_vid'],
         "sd_vid":  json_f['sd_vid'],
         "org_hd_vid":  json_f['org_hd_vid'],
         "org_sd_vid":  json_f['org_sd_vid'],
         "device":  json_f['cam_id'],
         "dur": float(json_f['event_duration']),
         "max_peak": float(json_f['peak_magnitude'])
      }
   }

# Get new stars info from an old JSON version 
def get_new_stars(json_f):
   new_stars = []
   if "cal_params" in json_f:
      if "cat_image_stars" in json_f['cal_params']:
         for star in json_f['cal_params']['cat_image_stars']:
            new_stars.append({
            "name": star[0],
            "mag": float(star[1]),
            "ra": float(star[2]),
            "dec": float(star[3]),
            "dist_px": float(star[6]),
            "i_pos": [float(star[7]),float(star[8])],
            "cat_dist_pos": [float(star[11]),float(star[12])],
            "cat_und_pos": [float(star[13]),float(star[14])]
            })

   return {"stars": new_stars}

# Get new frames from an old JSON Version
def get_new_frames(json_f):
   new_frames = []
   for frame in json_f['meteor_frame_data']:
      new_frames.append({
               "fn": int(frame[1]),
               "dt": frame[0], 
               "x":  int(frame[2]),
               "y":  int(frame[3]),
               "az": float(frame[9]),
               "el": float(frame[10]),
               "dec": float(frame[8]),
               "ra": float(frame[7]),
               "w": int(frame[4]),
               "h": int(frame[5]),
               "max_px": int(frame[6]) 
      })
   return {"frames": new_frames}

# Convert a whole old JSON file following the new DTD
def convert_json(json_file_path, sd_video_file_path, hd_video_file_path):
   
   # Load the initial JSON
   json_f = load_json_file(json_file_path)

   # Do we have a -reduced file?
   meteor_reduced_file = json_file_path.replace(".json", "-reduced.json")
   if(cfe(meteor_reduced_file)):
      reduced_info = load_json_file(meteor_reduced_file)
   else:
      print("ONLY REDUCED DETECTION CAN BE CONVERTED - reduce.json not found")
      sys.exit(0)

   # Analyse the json name
   analysed_name = old_name_analyser(json_file_path)
   
   # Get the device name if it doesn't exists in the JSON
   if('station_id' not in analysed_name):
      # We get the station id from what??,
      analysed_name['station_id'] = get_station_id()
      json_f['station_id'] = analysed_name['station_id']

   # Add the cam id to json_f
   if('cam_id' in analysed_name):
      json_f['cam_id'] = analysed_name['cam_id']
 
   # Add event duration to json_f
   if('event_duration' not in json_f and reduced_info is not None):
      json_f['event_duration'] = reduced_info['event_duration']
   
   # Add peak_magnitude duration to json_f
   if('peak_magnitude' not in json_f and reduced_info is not None):
      json_f['peak_magnitude'] = reduced_info['peak_magnitude']


   # Add the videos to json_f
   json_f['org_hd_vid'] = hd_video_file_path
   json_f['org_sd_vid'] = sd_video_file_path

   # Temporary until we move the videos
   json_f['hd_vid'] = hd_video_file_path
   json_f['sd_vid'] = sd_video_file_path

   # Convert info 
   info = get_new_info(json_f)

   # Add the original name with trim in case there's an issue
   info['info']['org_file_name'] = json_file_path

   calib = get_new_calib(reduced_info)
   stars = get_new_stars(reduced_info)

   # Get the frames here (from reduced_info)
   frames = get_new_frames(reduced_info)


   # The stars belong to calib
   calib['calib']['stars'] = stars['stars']
   
   return {"info": info['info'],"calib": calib['calib'],"frames": frames['frames']}


# Move new JSON file and  video files to meteor_archive
# in the proper folder
# from a old -reduced.json file
def move_old_detection_to_archive(json_file_path, sd_video_file_path, hd_video_file_path, display=True):
   
   # DEBUG
   cgitb.enable();

   # Get the new JSON file based on all info
   new_json_file = convert_json(json_file_path, sd_video_file_path, hd_video_file_path)
 
   # We fix the old name to get the proper info
   fixed_json_file_path = fix_old_file_name(json_file_path)

   # Get the closest param files
   param_files = get_active_cal_file(fixed_json_file_path)
   
   if(cfe(param_files[0][0])==0):
      print("PARAM FILES " + param_files[0][0]  + " not found" )
      sys.exit(0)
   #else:
   #   print("PARAM FILE: " + param_files[0][0])

   # Here we try to sync the HD and the SD files
   sync_res = False
   if(cfe(hd_video_file_path) and cfe(sd_video_file_path) and cfe(json_file_path.replace('.json','-reduced.json'))):
      sync_res = sync_hd_frames(hd_video_file_path,sd_video_file_path,json_file_path.replace('.json','-reduced.json'))
      if(sync_res == False):
         print("HD & SD video not synchronized!")

   new_json_file['calib']['org_file'] = param_files[0][0]

   # Determine the folder where to put the files
   tan = old_name_analyser(sd_video_file_path)
   tan['name'] = tan['name'].replace('.mp4','.json')
   tan['name'] = tan['name'].replace('-SD','') # Eventually
   new_folder = get_new_archive_folder(old_name_analyser(json_file_path))

   # If the new_folder doesn't exist, we create it
   if not os.path.exists(new_folder):
      os.makedirs(new_folder)

   # We move the videos to the folder
   if(cfe(hd_video_file_path)):
      new_hd_video_file = new_folder + tan['name'].replace(".json","-HD.mp4")
      shutil.copy2(hd_video_file_path,new_hd_video_file)

   if(cfe(sd_video_file_path)):
      new_sd_video_file = new_folder + tan['name'].replace(".json","-SD.mp4")
      shutil.copy2(sd_video_file_path,new_sd_video_file)   
   
   if(display is True):
      print("VIDEOS FILE SAVE TO " + new_folder)

 
   # Create the definitive json_content
   json_content = {}
   json_content['calib']   = new_json_file['calib']
   json_content['info']    = new_json_file['info']
   json_content['frames']  = new_json_file['frames']

   # Add the sync SD/HD if we have them
   if(sync_res != False): 
      json_content['sync']  = sync_res
   
   # Save the new JSON file
   save_json_file(new_folder + tan['name'], json_content)
  
   if(display is True):
      print("JSON SAVED TO " + new_folder + tan['name'])
   
   return new_folder + tan['name'],new_hd_video_file,new_sd_video_file



# Move detection to Archives
# and open the related reduce2 page
def move_to_archive(form):

   # DEBUG
   cgitb.enable();

   hd_video = form.getvalue("video_file")
   sd_video = form.getvalue("sd_video")
   json_file = form.getvalue("json_file")
 
 
   if(hd_video is None or cfe(hd_video)==0):
      print("HD video is missing.")
      sys.exit(0)

   if(sd_video is None or cfe(sd_video)==0):
      print("SD video is missing.")
      sys.exit(0)

   if(json_file is None or cfe(json_file)==0):
      print("JSON is missing.")   
      sys.exit(0)

   
   new_json,new_hd_vid,new_sd_vid = move_old_detection_to_archive(json_file,sd_video,hd_video, False)
   redirect_to("/pycgi/webUI.py?cmd=reduce2&video_file=" + new_hd_vid + "&clear_cache=1&c=" + str(random.randint(0,100000000)), "reduction")
   #print("/pycgi/webUI.py?cmd=reduce2&video_file=" + new_hd_vid + "&clear_cache=1&c=" + str(random.randint(0,100000000)))
   