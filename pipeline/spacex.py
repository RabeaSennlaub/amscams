#!/usr/bin/python3
import datetime
from lib.PipeUtil import cfe, save_json_file
import os
import glob
# script to archive footage from longer events like space-x rockets or satellites

ams_id = "AMS15"
# define cams that caught the event
cams = ['010087', '010086']

event_start_time = '2021_03_14_10_09_01'
event_end_time = '2021_03_14_10_11_01'
base_file = event_start_time[:-5]
date = event_start_time[0:10]
(year,month,day,hour,minute,second) = event_start_time.split("_" )
sd_dir = "/mnt/ams2/SD/proc2/" + date + "/" 
hd_dir = "/mnt/ams2/HD/"
out_dir = "/mnt/ams2/CUSTOM/SPACEX/2021_03_14/"
cloud_dir = "/mnt/archive.allsky.tv/" + ams_id + "/CUSTOM/SPACEX/2021_03_14/"
if cfe(out_dir,1) == 0:
   os.makedirs(out_dir)
if cfe(cloud_dir,1) == 0:
   os.makedirs(cloud_dir)


start_dt = datetime.datetime.strptime(event_start_time, "%Y_%m_%d_%H_%M_%S")
end_dt = datetime.datetime.strptime(event_end_time, "%Y_%m_%d_%H_%M_%S")

min_diff = int((end_dt - start_dt).total_seconds() / 60)
all_sd = []
all_hd = []
for i in range(0, min_diff+1):
   mn = int(minute) + i
   wild = base_file + str(mn) + "*.mp4"
   print("B:", wild)
   sd_files = glob.glob(sd_dir + wild)
   for ff in sd_files:
      if "trim" not in ff:
         all_sd.append(ff)
   sd_files = glob.glob(hd_dir + wild)
   for ff in sd_files:
      if "trim" not in ff:
         all_hd.append(ff)

html = ""
for vf in all_sd:
   fn = vf.split("/")[-1]
   out_file = ams_id + "_" + "SD_" + fn
   cmd = "cp " + vf + " " + out_dir + out_file
   html += "<a href=" + out_file + ">" + out_file + "</a><br>\n"
   if cfe(out_dir + out_file) == 0:
      print(cmd)
      os.system(cmd)
   if cfe(cloud_dir + out_file) == 0:
      cmd = "cp " + vf + " " + cloud_dir + out_file
      print(cmd)
      os.system(cmd)

for vf in all_hd:
   fn = vf.split("/")[-1]
   out_file = ams_id + "_" + "HD_" + fn
   cmd = "cp " + vf + " " + out_dir + out_file
   html += "<a href=" + out_file + ">" + out_file + "</a><br>\n"
   if cfe(out_dir + out_file) == 0:
      print(cmd)
      os.system(cmd)
   if cfe(cloud_dir + out_file) == 0:
      cmd = "cp " + vf + " " + cloud_dir + out_file
      print(cmd)
      os.system(cmd)
data = {}
data['sd_files'] = all_sd
data['hd_files'] = all_hd
save_json_file(out_dir + "files.json", data)
out = open(out_dir + "index.html", "w")
out.write(html)
os.system("cp " + out_dir + "index.html" + " " + cloud_dir)
print(out_dir)
print(cloud_dir)
