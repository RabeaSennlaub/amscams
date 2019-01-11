import math
import cv2
import random
import numpy as np
from detectlib import eval_cnt
from scipy import signal
from scipy.interpolate import splrep, sproot, splev
from pathlib import Path
import json

import brightstardata as bsd
mybsd = bsd.brightstardata()
bright_stars = mybsd.bright_stars

def calc_dist(p1,p2):
   x1,y1 = p1
   x2,y2 = p2
   dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
   return dist


def save_json_file(json_file, json_data):
   with open(json_file, 'w') as outfile:
      json.dump(json_data, outfile)
   outfile.close()

def load_json_file(json_file):
   with open(json_file, 'r') as infile:
      json_data = json.load(infile)
   return(json_data)

def load_stars(cal_file):
   star_file = cal_file.replace(".jpg", "-stars.txt")
   sf = open(star_file, "r")
   starlist = []
   for line in sf:
      line = line.replace("\n", "")
      line = line.replace(" Mike Star: ", "")
      data = line.split(" ")
      if len(data) == 5:
         star_name = data[1]
         star_name = star_name.replace(" at ", "")
         star_img_x = data[3]
         star_img_x = star_img_x.replace("(", "")
         star_img_x = star_img_x.replace(",", "")
         star_img_y = data[4]
         star_img_y = star_img_y.replace(")", "")
         starlist.append((star_name, star_img_x, star_img_y))
   return(starlist)


def bound_cnt(x,y,img_w,img_h):
   sz = 10 

   if x - sz < 0:
      mnx = 0
   else:
      mnx = x - sz 

   if y - sz < 0:
      mny = 0
   else:
      mny = y - sz 

   if x + sz > img_w - 1:
      mxx = img_w - 1 
   else: 
      mxx = x + sz

   if y + sz > img_h -1:
      mxy = img_h - 1
   else:
      mxy = y + sz
   return(mnx,mny,mxx,mxy)


def clean_star_bg(cnt_img, bg_avg):
   for x in range(0,cnt_img.shape[1]):
      for y in range(0,cnt_img.shape[0]):
         px_val = cnt_img[y,x]
         if px_val < bg_avg:
            cnt_img[y,x] = random.randint(int(bg_avg - 3),int(bg_avg+3))
   return(cnt_img)
 

def find_stars(med_stack_all): 
   img_height, img_width= med_stack_all.shape
   max_px = np.max(med_stack_all)
   avg_px = np.mean(med_stack_all)
   med_cpy = med_stack_all.copy()
   pdif = max_px - avg_px
   pdif = int(pdif / 4) + avg_px
   bg_avg = 0
   _, star_bg = cv2.threshold(med_stack_all, pdif, 255, cv2.THRESH_BINARY)
   thresh_obj= cv2.convertScaleAbs(star_bg)
   (_, cnts, xx) = cv2.findContours(thresh_obj.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   star_pixels = []
   non_star_pixels = []
   cloudy_areas = []
   for (i,c) in enumerate(cnts):
      x,y,w,h= cv2.boundingRect(cnts[i])
      if w < 30 and h < 30:
         cnt_img = med_stack_all[y:y+h,x:x+w]
         (max_px, avg_px,px_diff,max_loc) = eval_cnt(cnt_img)

         mx,my = max_loc
         cx = x + mx
         cy = y + my
         mnx,mny,mxx,mxy = bound_cnt(cx,cy,img_width,img_height)
         cnt_img = med_stack_all[mny:mxy,mnx:mxx]
         cnt_w,cnt_h = cnt_img.shape
         if cnt_w > 0 and cnt_h > 0:
            is_star = star_test(cnt_img)
            if is_star == 1:
               bg_avg = bg_avg + np.mean(cnt_img)
               star_pixels.append((cx,cy))
               cv2.circle(med_cpy, (int(cx),int(cy)), 5, (255,255,255), 1)
               #cv2.imshow('pepe', cnt_img)
               #cv2.waitKey(0)
            else:
               cv2.rectangle(med_cpy, (cx-5, cy-5), (cx + 5, cy + 5), (255, 0, 0), 1)
               non_star_pixels.append((cx,cy))
      else:
         cloudy_areas.append((x,y,w,h))
         cv2.rectangle(med_cpy, (x, y), (x + w, y + w), (255, 0, 0), 3)

   if len(non_star_pixels) > 0 and len(star_pixels) > 0:
      perc_cloudy = int(len(non_star_pixels) / len(star_pixels))  * 100
      desc = str(len(star_pixels)) + " stars " + str(len(non_star_pixels)) + " non stars " + str(perc_cloudy) + "% cloudy"

   else :
      perc_cloudy = 0
      perc_clear = 100
      desc = str(len(star_pixels)) + " stars " + str(len(non_star_pixels)) + " non stars " + str(perc_clear) + "% clear"

   status = ""

   if len(star_pixels) > 10:
      status = "clear"
   if len(non_star_pixels) > len(star_pixels) or len(star_pixels) <= 5:
      status = "cloudy"
   if len(non_star_pixels) == 0 and len(star_pixels) == 0:
      status = "cloudy"
   if len(non_star_pixels) >= 5 and len(star_pixels) <= 5:
       status = "partly cloudy "

   cv2.putText(med_cpy, str(status),  (10,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
   cv2.imshow('pepe', med_cpy)
   cv2.waitKey(10)
   return(status,star_pixels, non_star_pixels, cloudy_areas)

def find_bright_pixels(med_stack_all, solved_file):
   img_height,img_width = med_stack_all.shape
   med_cpy = med_stack_all.copy()
   star_pixels = []
   max_px = np.max(med_stack_all)
   avg_px = np.mean(med_stack_all)
   pdif = max_px - avg_px
   pdif = int(pdif / 15) + avg_px
   pdif = avg_px + 10 
   #star_bg = 255 - cv2.adaptiveThreshold(med_stack_all, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
  
   _, star_bg = cv2.threshold(med_stack_all, pdif, 255, cv2.THRESH_BINARY)

   #cv2.imshow('pepe', star_bg)
   #cv2.waitKey(0)

   #star_bg = cv2.GaussianBlur(star_bg, (7, 7), 0)
   thresh_obj = cv2.dilate(star_bg, None , iterations=4)
   #thresh_obj= cv2.convertScaleAbs(star_bg)
   (_, cnts, xx) = cv2.findContours(thresh_obj.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   masked_pixels = []

   for (i,c) in enumerate(cnts):
      x,y,w,h = cv2.boundingRect(cnts[i])
      cv2.rectangle(star_bg, (x, y), (x + w+5, y + h+5), (255, 0, 0), 1)
   #cv2.imshow('pepe', star_bg)
   #cv2.waitKey(1)

   bg_avg = 0

   for (i,c) in enumerate(cnts):
      x,y,w,h = cv2.boundingRect(cnts[i])
      if True:
         cnt_img = med_stack_all[y:y+h,x:x+w]
         (max_px, avg_px,px_diff,max_loc) = eval_cnt(cnt_img)

         mx,my = max_loc
         cx = x + mx
         cy = y + my
         mnx,mny,mxx,mxy = bound_cnt(cx,cy,img_width,img_height) 
         cnt_img = med_stack_all[mny:mxy,mnx:mxx]
         cnt_w,cnt_h = cnt_img.shape
         if cnt_w > 0 and cnt_h > 0:
            is_star = star_test(cnt_img)
            if is_star == 1:
               bg_avg = bg_avg + np.mean(cnt_img)
               star_pixels.append((cx,cy))
               cv2.circle(med_cpy, (int(cx),int(cy)), 5, (255,255,255), 1)
               #cv2.imshow('pepe', cnt_img)
               #cv2.waitKey(0)
            else:
               cv2.rectangle(med_cpy, (cx-5, cy-5), (cx + 5, cy + 5), (255, 0, 0), 1)
         else:
               cv2.rectangle(med_cpy, (cx-15, cy-15), (cx + 15, cy + 15), (255, 0, 0), 1)

   if len(star_pixels) > 0:
      bg_avg = bg_avg / len(star_pixels) 
   else:
      bg_avg = 35

   file_exists = Path(solved_file)
   if file_exists.is_file() is False:
      # if plate image does not exist! 
      plate_image = med_stack_all.copy()
      for x in range(0,plate_image.shape[1]):
         for y in range(0,plate_image.shape[0]):
            plate_image[y,x] = random.randint(int(0),int(10))

      star_sz = 10
      for star in star_pixels:
         sx,sy = star
         mnx,mny,mxx,mxy = bound_cnt(sx,sy,img_width,img_height) 
         star_cnt = med_stack_all[mny:mxy,mnx:mxx]
         star_cnt = clean_star_bg(star_cnt, bg_avg)
         plate_image[mny:mxy,mnx:mxx] = star_cnt

   else:
      plate_image = cv2.imread(solved_file, 0)
 



   return(star_pixels, plate_image)

def star_test(cnt_img):
   PX = []
   PY = []
   ch,cw = cnt_img.shape
   my = int(ch / 2)
   mx = int(cw / 2)
   max_px = np.max(cnt_img)
   avg_px = np.mean(cnt_img)
   px_diff = max_px - avg_px 

   for x in range(0,cw-1):
      px_val = cnt_img[my,x]
      PX.append(px_val)
      #cnt_img[my,x] = 255
   for y in range(0,ch-1):
      py_val = cnt_img[y,mx]
      PY.append(py_val)
      #cnt_img[y,mx] = 255

   ys_peaks = signal.find_peaks(PY)
   y_peaks = len(ys_peaks[0])
   xs_peaks = signal.find_peaks(PX)
   x_peaks = len(xs_peaks[0])


   #cv2.imshow('pepe', cnt_img)
   #cv2.waitKey(0)
   if px_diff > 10:
      is_star = 1 
   else:
      is_star = 0

   return(is_star)

def astr_line(line):
   line = line.replace("\n", "")
   line = line.replace("  Mike Star: ", "")
   el = line.split("at")
   if "at" not in line:
      exit()

   if len(el) == 2:
      cname,num = el 
   else:
      print("fuck:", line)
      exit()
   num = num.replace(" ","")
   num = num.replace("(","")
   num = num.replace(")","")
   fields = line.split(" ")
   if "(" in cname:
      common_name, cat_name = cname.split("(")
      cname = cat_name
      cname = cname.replace(")", "")
      #cname.replace(cat_name, "")
   else:
      common_name = ""
   cname = cname.rstrip()
   common_name = common_name.rstrip()

   ast_x,ast_y = num.split(",")
   return(cname,common_name,ast_x,ast_y)

def parse_astr_star_file(star_data_file, limit_mag_start = -10, limit_mag_end = 3):
   bright_stars_found = []
   bright_stars_notfound = []
   fp = open(star_data_file, "r")
   for line in fp:
      status = 0
      bname = ""
      common_name = ""
      star_name, common_name, ast_x, ast_y = astr_line(line)

      (status, bname, cons, ra,dec,mag) = find_star_by_name(star_name, common_name)

      if int(status) == 1:
         data = bname + "," + cons + "," + str(ra) + "," + str(dec) + "," + str(mag) + "," + str(ast_x) + "," + str(ast_y)
         data = (bname, cons,ra,dec,mag,ast_x,ast_y)
         if limit_mag_start <= float(mag) <= limit_mag_end:
            bright_stars_found.append(data)
      else:
         data = (star_name,0,0,0,ast_x,ast_y)
         bright_stars_notfound.append(data)

   return(bright_stars_found, bright_stars_notfound)

def find_star_by_name(star_name, common_name, debug = 1):
   for bname, cname, ra, dec, mag in bright_stars:
      dbname = bname.decode("utf-8")
      dcname = cname.decode("utf-8")
      if star_name in dbname and star_name != "" and dbname != "":  
         return(1,dbname, dcname, ra,dec,mag)
      if common_name in dcname and common_name != "" and dcname != "":
         return(1,dbname, dcname, ra,dec,mag)
   print("Could not find", star_name, common_name, "in catalog!")
   exit()
   return(0,0,0,0,0,0)

