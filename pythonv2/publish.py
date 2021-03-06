#!/usr/bin/python3

"""
This script creates all the html pages related to a fireball event. 
  * Run this script through the day to keep data up to date and in sync.
  * Run this script after a day has finished to close out all work relating to that day. 
  * Script will perform the following functions.
     - Create an html page "report" for a given camera of a station
"""

import os
import glob
import sys
import json
import re
import random

from lib.FileIO import *  
from lib.Video_Tools_Fundamentals import get_ROI_from_arc_json

from doDay import analyse_report_file

REGEX_JSON_FROM_CLOUD = r"\/(\w*)\/METEOR\/(\d{4})\/(\d{2})\/(\d{2})\/\d{4}_\d{2}_\d{2}_(\d{2})_(\d{2})_(\d{2})_(\d{3})_(\w{6})-trim(\d{4}|\d{3}|\d{2}|\d{1}).json"
REGEX_JSON_FROM_CLOUD2 = r"\/(\w*)\/METEOR\/(\d{4})\/(\d{2})\/(\d{2})\/\d{4}_\d{2}_\d{2}_(\d{2})_(\d{2})_(\d{2})_(\d{3})_(\w{6})-trim-(\d{4}|\d{3}|\d{2}|\d{1}).json"
REGEX_GROUP_JSON_FROM_CLOUD = ["all_path","station_id","year","month","day","hour","min","sec","ms","cam_id","trim"]
 
#PATH TO CLOUD ARCHIVES
PATH_TO_CLOUD = "/mnt/archive.allsky.tv"
#PATH_TO_CLOUD = "/mnt/ams2/meteor_archive"

# TEMPLATES
OBSERVER_REPORT_TEMPLATE = "/home/ams/amscams/pythonv2/templates/allsky.tv.obs_report.html"

 
# ARCHIVE URL
ARCHIVE_URL= "http://archive.allsky.tv"  


# Analyse the json file names
def analyse_event_json_file(file_name):
   if "-trim-" in file_name:
      matches = re.finditer(REGEX_JSON_FROM_CLOUD2, file_name, re.MULTILINE)
   else:
      matches = re.finditer(REGEX_JSON_FROM_CLOUD, file_name, re.MULTILINE)
   res = {}
   print("FILE NAMES:", file_name)  
   for matchNum, match in enumerate(matches, start=1):
      print("MATCES:", match)  
      for groupNum in range(0, len(match.groups())): 
         if(match.group(groupNum) is not None):
            res[REGEX_GROUP_JSON_FROM_CLOUD[groupNum]] = match.group(groupNum)
         groupNum = groupNum + 1

   return res



def make_event_station_report(json_file):

   # Format of the JSON file:
   # /AMS7/METEOR/2019/12/24/2019_12_24_08_17_10_000_010041-trim1298.json
   # Format of the OUTPUT HTML file
   # /AMS7/METEOR/2019/12/24/2019_12_24_08_17_10_000_010041-trim1298.html
  
   # We load (and test) the json
   json_data = load_json_file(PATH_TO_CLOUD+json_file)
   if(json_data is False): 
      print(PATH_TO_CLOUD+json_file + " not found")
      sys.exit(0)

   # Build the page based on template  
   with open(OBSERVER_REPORT_TEMPLATE, 'r') as file:
      template = file.read()
    
   # Add station id & other static info on the title
   analysed_name = analyse_event_json_file(json_file) 
 
   # Create link to daily report
   link_to_daily_report = "/"+analysed_name['station_id']+"/REPORTS/"+analysed_name['year']+"/"+analysed_name['month']+"_"+analysed_name['day']+"/index.html"
   full_date = analysed_name['year']+'/'+analysed_name['month']+'/'+analysed_name['day']+' '+analysed_name['hour']+":"+analysed_name['min']+":"+analysed_name['sec']+"."+analysed_name['ms']



   # Template data
   template = template.replace('{STATION_ID}',analysed_name['station_id'])
   template = template.replace('{CAM_ID}',analysed_name['cam_id'])
   template = template.replace('{DATE}',full_date)
   template = template.replace('{DAY}',analysed_name['year']+'/'+analysed_name['month']+'/'+analysed_name['day'])
   template = template.replace('{TIME}',analysed_name['hour']+':'+analysed_name['min']+':'+analysed_name['sec']+'.'+analysed_name['ms'])
   template = template.replace('{LINK_TO_DAILY_REPORT}',link_to_daily_report)

   # Get the first frame #
   # to optimize the JS on the obs report (the frames are getting from js)
   first_frame = 0
   how_many_frames = 0
   all_fn = []
   if('frames' in json_data):
      for f in json_data['frames']:
         all_fn.append(f['fn'])
      first_frame = min(all_fn)
      how_many_frames = len(json_data['frames'])

   template = template.replace('{HOW_MANY_FRAMES}',str(how_many_frames))
   template = template.replace('{FIRST_FRAME}',str(first_frame))
 
   # Get (HD) Video Path
   HD_vid = True
   video_btn = ""
   sd_video_full_path = ""
   hd_video_full_path =  PATH_TO_CLOUD + json_file.replace('.json','-HD.mp4')

   if(cfe(hd_video_full_path)==0):
      sd_video_full_path = hd_video_full_path.replace('-HD','-SD')
      if(cfe(sd_video_full_path)==0):
         hd_video_full_path = ""
      else:
         HD_vid = False 

   if(HD_vid== True ): 
      video_btn += '<a class="col btn btn-secondary mt-0 mb-0 ml-1 vid-link d-block" href="'+json_file.replace('.json','-HD.mp4')+'"><i class="icon-youtube"></i> HD Video</a>'
   
   if(cfe(sd_video_full_path)!=0):
      video_btn += '<a class="col btn btn-secondary mt-0 mb-0 ml-1 vid-link d-block" href="'+json_file.replace('.json','-SD.mp4')+'"><i class="icon-youtube"></i> SD Video</a>'
  
   # Do we have a cropped version of the video?
   cropped_hd_full_path = hd_video_full_path.replace('-HD','-HD-cropped') 
   we_have_a_cropped = False
   real_cropped = ''

   if(cfe(cropped_hd_full_path)==1):
      real_cropped = cropped_hd_full_path.replace(PATH_TO_CLOUD,ARCHIVE_URL)
      template = template.replace('{CROPPED_VIDEO}',real_cropped) 
      we_have_a_cropped = True
   else:
      cropped_sd_full_path = sd_video_full_path.replace('-SD','-SD-cropped')
      if(cfe(cropped_sd_full_path)==1):
         real_cropped = cropped_sd_full_path.replace(PATH_TO_CLOUD,ARCHIVE_URL)
         template = template.replace('{CROPPED_VIDEO}', real_cropped)
         we_have_a_cropped = True

    
   if( we_have_a_cropped == True):
      # Add info about the cropped video
      # 1) Size
      x,y,w,h  = get_ROI_from_arc_json(PATH_TO_CLOUD+json_file)

      template = template.replace('{X}',str(x)) 
      template = template.replace('{Y}',str(y)) 
      template = template.replace('{W}',str(w-x))
      template = template.replace('{H}',str(h-y)) 
 

   else:
      template = template.replace('{X}','') 
      template = template.replace('{Y}','') 
      template = template.replace('{W}','') 
      template = template.replace('{H}','') 

 

   # Frame by Frame animation button
   video_btn += '<a id="play_anim_tv" class="col btn btn-secondary mt-0 mb-0 ml-1 d-block disabled" href="#"><i class="icon-youtube"></i> Frames by F</a>'
 
   # Add the video buttons
   if(video_btn!=''):
      template = template.replace('{VIDEO_BTNS}',video_btn)
   else:
      template = template.replace('{VIDEO_BTNS}','')
    
   # Video Btns
   playBtns = '''
      <div class="video-controls-holder d-flex justify-content-between">
      <ul id="video-controls" class="controls">
			<li><button id="playpause" type="button" class="btn btn-secondary btn-sm"><span class="icon-play1"></span></button></li>
			<li><button id="stop" type="button" class="btn btn-secondary btn-sm"><span class="icon-stop"></span></button></li>
      </ul>
      <div class="rtp">
         <small class="d-block">Stars</small><input type="range" value="100" name="stars_transp" max="100" min="0">
      </div>
      <div class="rtp mr-4">
         <small class="d-block">Meteor Path Frames</small><input type="range" value="100" name="frame_transp" max="100" min="0">
      </div> 
    '''

   template = template.replace('{VIDEO}','<figure id="videoContainer" data-fullscreen="false"><video id="main_video_player" width="960" height="540" loop=""><source src="'+json_file.replace('.json','-HD.mp4')+'" type="video/mp4"></video>'+playBtns+'</figure>')
  
   # NO-Cache
   template = template.replace("{RAND}",str(random.randint(0, 99999999)))

   # JSON File
   template = template.replace("{JSON_FILE}",ARCHIVE_URL+json_file)

   #JSON DATA
   template = template.replace("{JSON_DATA}",json.dumps(json_data))

   # DETECTION DETAILS
   report_details = ''

   if('report' in json_data):
      report_details += '<dt class="col-4">Date &amp; Time</dt><dd class="col-8">'+full_date+'s</dd>'
      if('dur' in json_data['report']):
         report_details += '<dt class="col-4">Duration</dt><dd class="col-8"><span id="dur">'+str(json_data['report']['dur'])+'</span>s</dd>'
      if('max_peak' in json_data['report']):
         report_details += '<dt class="col-4">Max Intensity</dt><dd class="col-8">'+str(json_data['report']['max_peak'])+'</dd>'
      if('angular_vel' in json_data['report']):
         report_details += '<dt class="col-4">Ang. Velocity</dt><dd class="col-8">'+str(json_data['report']['angular_vel'])+'&deg;/sec</dd>'
      if('point_score' in json_data['report']):
            pts = str(json_data['report']['point_score'])
            if(json_data['report']['point_score']>3):
               pts = "<b style='color:#f00'>"+ pts +  "</b>"
            report_details += '<br/><dt class="col-4">Point Score</dt><dd class="col-8" id="point_score_val">'+pts+'</dd>'

   if('calib' in json_data):
      if('device' in json_data['calib']):
         if('total_res_px' in json_data['calib']['device']):
            pts = str(json_data['calib']['device']['total_res_px'])
            
            # Not red if >3
            #            if(json_data['calib']['device']['total_res_px']>3):
            #   pts = "<b style='color:#f00'>"+ pts +  "</b>"
            report_details += '<dt class="col-4">Res. Error</dt><dd class="col-8">'+pts+'</dd>'
    
   # Report Details
   template = template.replace("{REPORT_DETAILS}",report_details)



   # Get the full-cropped (for the ROI)
   # ex:
   #http://archive.allsky.tv/AMS7/METEOR/2019/12/24/2019_12_24_06_09_13_000_010042-trim0167.json
   #>
   #http://archive.allsky.tv/AMS7/DETECTS/PREVIEW/2019/2019_12_24/2019_12_24_06_09_13_000_010042-trim0167-prev-full.jpg
   full_cropped = ARCHIVE_URL + '/' + analysed_name['station_id']  +  '/'  +  "DETECT"  +  '/'  + "PREVIEW"  +  '/'  + analysed_name['year']  +  '/'  + json_file.replace('.json','-prev-full.jpg')
   template = template.replace("{CROPPED_FULL}",full_cropped)

   
  
   # Create Template
   f = open(PATH_TO_CLOUD+json_file.replace('.json','.html'), "w+")
   f.write(template)
   f.close() 

   print(json_file.replace('.json','.html') +  " created.")
 


## Make Event Report Page
def make_event_report(json_file):
   print("MAKE EVENT REPORT")


### COMMAND
cmd = sys.argv[1]

if cmd == "event_station_report":
   make_event_station_report(sys.argv[2])
