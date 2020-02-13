
import os
import ephem
import glob
import re
import sys
import json

from lib.Get_Cam_position import get_device_position
from lib.Get_Station_Id import get_station_id
from lib.Get_Cam_ids import get_the_cam_ids

MINUTE_FOLDER = '/mnt/ams2/SD/proc2'
IMAGES_MINUTE_FOLDER = 'images'
DEFAULT_HORIZON_EPHEM = '-0:34'
DEFAULT_PRESSURE = 0

# Minute stacks regex
MINUTE_STACK_EXT = "stacked-tn"
MINUTE_FILE_NAMES_REGEX = r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})_(\w{6})-stacked-tn.(\w{3})"
MINUTE_FILE_NAMES_REGEX_GROUP = ["full","year","month","day","hour","min","sec","ms","cam_id","ext"]

# Parses a regexp (MINUTE_FILE_NAMES_REGEX) a minute file name
# and returns all the info defined in MINUTE_FILE_NAMES_REGEX_GROUP
def minute_name_analyser(file_name):
   matches = re.finditer(MINUTE_FILE_NAMES_REGEX, file_name, re.MULTILINE)
   res = {}
    
   for matchNum, match in enumerate(matches, start=1):
      for groupNum in range(0, len(match.groups())): 
         if(match.group(groupNum) is not None):
            res[MINUTE_FILE_NAMES_REGEX_GROUP[groupNum]] = match.group(groupNum)
         groupNum = groupNum + 1
 
   return res
  
# Get sun az & alt to determine if it's a daytime or nightime minute
def get_sun_details(capture_date):

   device_position = get_device_position()

   if('lat' in device_position and 'lng' in  device_position):

      obs = ephem.Observer()

      obs.pressure = DEFAULT_PRESSURE
      obs.horizon = DEFAULT_HORIZON_EPHEM
      obs.lat  = device_position['lat']
      obs.lon  = device_position['lng']
      obs.date = capture_date

      sun = ephem.Sun()
      sun.compute(obs)

      (sun_alt, x,y) = str(sun.alt).split(":")
      saz = str(sun.az)
      (sun_az, x,y) = saz.split(":")
      if int(sun_alt) < -1:
         sun_status = "n"   # Night
      else:
         sun_status = "d"   # Day

      return sun_az,sun_alt,sun_status
   
   else:
      return 0,0,"?"


# Create index for a given year
def create_json_index_minute_day(day,month, year):

   # Main dir to glob
   main_dir = MINUTE_FOLDER +  os.sep + str(year) + '_' + str(month).zfill(2) + '_' + str(day).zfill(2) + os.sep + IMAGES_MINUTE_FOLDER
 
   index_day = {'station_id':get_station_id(),'year':int(year),'months':int(month),'day':int(day),'hours':[]}


   cam_ids = get_the_cam_ids();

   stacks_per_minute = []

   

   for i in range(0,24):
      stacks_per_minute.append(i)
      for cam_id in cam_ids:
         stacks_per_minute[i] = {"cam_id": cam_id, "stacks": []}

   print(stacks_per_minute)
   sys.exit(0)

 
   for minute_stack in sorted(glob.iglob(main_dir + '*' + os.sep + '*' + MINUTE_STACK_EXT + '*', recursive=True), reverse=True):	
      
      cur_stack_data = []

      # We analyse the name
      analysed_minute = minute_name_analyser(minute_stack) 

      # Get Sun details at the date of the capture
      sun_az,sun_alt,sun_status = get_sun_details(analysed_minute['year']+'/'+analysed_minute['month']+'/'+analysed_minute['day']+' ' + analysed_minute['hour']+ ':' + analysed_minute['min']+ ':'+ analysed_minute['sec'])
 
      cur_stack_data =  {
          'f':minute_stack,
          't': analysed_minute['hour'] +':'+ analysed_minute['min'] +':'+ analysed_minute['sec'],
          'sun': {
             'az': float(sun_az),
             'alt': float(sun_alt),
             'status': sun_status
          }    
      }

      try: 
         stacks_per_minute[int(analysed_minute['min'])]
      except:
         stacks_per_minute[int(analysed_minute['min'])] = []
   
      stacks_per_minute[analysed_minute['min']].append(cur_stack_data)

   print(stacks_per_minute)

# Write index for a given day
def write_day_minute_index(day, month, year):
   json_data = create_json_index_minute_day(day,month, year)  

   # Write Index if we have data
   if('hours' in json_data): 
      output_dir = MINUTE_FOLDER +  os.sep + str(year) + os.sep + str(month).zfill(2) + '_' + str(day).zfill(2)

      # Just in case...
      if not os.path.exists(output_dir):
         os.makedirs(output_dir)

      with open(output_dir + os.sep + str(month).zfill(2) + '_' + str(day).zfill(2) + ".json", 'w') as outfile:
         #Write compress format
         json.dump(json_data, outfile)
 
      print(output_dir + os.sep + str(month).zfill(2) + '_' + str(day).zfill(2) + ".json - created")

      outfile.close() 
      return True
   
   return False