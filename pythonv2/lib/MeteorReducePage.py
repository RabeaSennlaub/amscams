import re
import cgitb

# PATTERN FOR THE FILE NAMES
# YYYY_MM_DD_HH_MM_SS_MSS_CAM_STATION[_HD].EXTENSION
FILE_NAMES_REGEX = r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})_(\d{6})_([^_^.]+)(_HD)?(\.)?(\.[0-9a-z]+$)"
FILE_NAMES_REGEX_GROUP = ["full","year","month","day","hour","min","sec","ms","cam_id","station_id","HD","ext"]


def name_analyser(file_names):
   matches = re.finditer(FILE_NAMES_REGEX, file_names, re.MULTILINE)
   res = {}
  
   for matchNum, match in enumerate(matches, start=1):
      
      for groupNum in range(0, len(match.groups())):
         
         
         if(match.group(groupNum) is not None):
            res[FILE_NAMES_REGEX_GROUP[groupNum]] = match.group(groupNum)
            print(str(groupNum) + " > " + match.group(groupNum))
         
         groupNum = groupNum + 1

   return res


# GENERATES THE REDUCE PAGE METEOR
# from a URL 
# cmd=reduce2
# &video_file=[VIDEO_FILE].mp4
def reduce_meteor2(json_conf,form):
   
   # Debug
   cgitb.enable()

   # Get Video File & Analyse the Name to get quick access to all info
   video_file    = form.getvalue("video_file")
   analysed_name = name_analyser(video_file)

   print(analysed_name)
   exit
   
   # Is it HD?
   if(analysed_name["HD"] is not None):
      HD = True
      meteor_json_file = video_file.replace("_HD.mp4", ".json") 
   else:
      HD = False
      meteor_json_file = video_file.replace(".mp4", ".json") 

   
   print(meteor_json_file)

