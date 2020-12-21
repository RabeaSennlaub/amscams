from flask import Flask, request
from FlaskLib.FlaskUtils import get_template
from lib.PipeUtil import cfe, load_json_file, save_json_file
from lib.PipeAutoCal import fn_dir
import time
import cv2
import os
import numpy as np

def detail_page(amsid, date, meteor_file):
   MEDIA_HOST = request.host_url.replace("5000", "80")
   MEDIA_HOST = ""
   METEOR_DIR = "/mnt/ams2/meteors/"
   METEOR_DIR += date + "/"
   METEOR_VDIR = METEOR_DIR.replace("/mnt/ams2", "")

   year,mon,day = date.split("_")
   base_name = meteor_file.replace(".mp4", "")
   json_conf = load_json_file("../conf/as6.json")
   obs_name = json_conf['site']['obs_name']
   CACHE_DIR = "/mnt/ams2/CACHE/" + year + "/" + mon + "/" + base_name + "/"
   CACHE_VDIR = CACHE_DIR.replace("/mnt/ams2", "")
   mjf = METEOR_DIR + meteor_file.replace(".mp4", ".json")
   mjvf = METEOR_VDIR + meteor_file.replace(".mp4", ".json")
   mjrf = METEOR_DIR + meteor_file.replace(".mp4", "-reduced.json")
   mjrvf = METEOR_VDIR + mjrf.replace("/mnt/ams2", "")
   if cfe(mjf) == 1:
      mj = load_json_file(mjf)
   else:
      return("meteor json not found.")

   sd_trim = meteor_file
   if "hd_trim" in mj:
      hd_trim,hdir  = fn_dir(mj['hd_trim'])
      hd_stack = hd_trim.replace(".mp4", "-stacked.jpg")
   else:
      hd_trim = None
      hd_stack = None

   sd_stack = sd_trim.replace(".mp4", "-stacked.jpg")
   half_stack = sd_stack.replace("stacked", "half-stack")
   if cfe(METEOR_DIR + half_stack) == 0:
      if cfe(METEOR_DIR + sd_stack) == 1:
         simg = cv2.imread(METEOR_DIR + sd_stack)
         simg  = cv2.resize(simg,(1920,1080))
         simg  = cv2.resize(simg,(960,540))
         cv2.imwrite(METEOR_DIR + half_stack,  simg)
         print("SAVED HALF", METEOR_DIR + half_stack, simg.shape)
      else:
         print("NO SD ", sd_stack)
 
   az_grid = ""
   header = get_template("FlaskTemplates/header.html")
   footer = get_template("FlaskTemplates/footer.html")
   nav = get_template("FlaskTemplates/nav.html")
   template = get_template("FlaskTemplates/meteor_detail.html")

   #footer = footer.replace("{RAND}", str(time.time()))
   if "location" in json_conf:
      template = template.replace("{LOCATION}", json_conf['site']['location'])
   else:
      template = template.replace("{LOCATION}", "")
   template = template.replace("{HEADER}", header)
   template = template.replace("{FOOTER}", footer)
   template = template.replace("{NAV}", nav)
   template = template.replace("{OBS_NAME}", obs_name)
   template = template.replace("{AMSID}", amsid)
   template = template.replace("{MEDIA_HOST}", MEDIA_HOST)
   template = template.replace("{HALF_STACK}", METEOR_VDIR + half_stack)
   if hd_stack is None or hd_stack == 0:
      template = template.replace("{HD_STACK}", "#")
   else:
      template = template.replace("{HD_STACK}", METEOR_VDIR + hd_stack)
   template = template.replace("{SD_STACK}", METEOR_VDIR + sd_stack)
   if hd_trim is None or hd_trim == 0:
      template = template.replace("{HD_TRIM}", "#")
   else:
      template = template.replace("{HD_TRIM}", METEOR_VDIR + hd_trim)
   template = template.replace("{AZ_GRID}", METEOR_VDIR + az_grid)
   template = template.replace("{JSON_CONF}", mjvf)
   template = template.replace("{METEOR_JSON}", mjvf)
   template = template.replace("{SD_TRIM}", METEOR_VDIR + sd_trim)
   template = template.replace("{METEOR_REDUCED_JSON}", mjrvf)

   if "best_meteor" not in mj: 
      template = template.replace("{START_TIME}", "-")
      template = template.replace("{DURATION}", "-")
      template = template.replace("{MAX_INTENSE}", "-")
      template = template.replace("{START_AZ}", "-")
      template = template.replace("{START_EL}", "-")
      template = template.replace("{END_EL}", "-")
      template = template.replace("{END_AZ}", "-")
      template = template.replace("{START_RA}", "-")
      template = template.replace("{END_RA}", "-")
      template = template.replace("{ANG_VEL}", "-")
      template = template.replace("{ANG_SEP}", "-")
      template = template.replace("{RA}", str("-"))
      template = template.replace("{DEC}", str("-"))
      template = template.replace("{AZ}", str("-"))
      template = template.replace("{EL}", str("-"))
      template = template.replace("{POSITION_ANGLE}", str("-"))
      template = template.replace("{PIXSCALE}", str("-"))
      template = template.replace("{IMG_STARS}", str("-"))
      template = template.replace("{CAT_STARS}", str("-"))
      template = template.replace("{RES_PX}", str("-"))
      template = template.replace("{RES_DEG}", str("-"))


   else:
      dur = len(mj['best_meteor']['ofns']) / 25 
      template = template.replace("{START_TIME}", mj['best_meteor']['dt'][0])
      template = template.replace("{DURATION}", str(dur)[0:4])
      template = template.replace("{MAX_INTENSE}", str(max(mj['best_meteor']['oint'])))

      template = template.replace("{START_AZ}", str(mj['best_meteor']['azs'][0])[0:5])
      template = template.replace("{END_AZ}", str(mj['best_meteor']['azs'][-1])[0:5])
      template = template.replace("{START_RA}", str(mj['best_meteor']['ras'][0])[0:5])
      template = template.replace("{END_RA}", str(mj['best_meteor']['ras'][-1])[0:5])
      template = template.replace("{START_DEC}", str(mj['best_meteor']['decs'][0])[0:5])
      template = template.replace("{END_DEC}", str(mj['best_meteor']['decs'][-1])[0:5])
      template = template.replace("{START_EL}", str(mj['best_meteor']['els'][0])[0:5])
      template = template.replace("{END_EL}", str(mj['best_meteor']['els'][-1])[0:5])
      template = template.replace("{ANG_VEL}", str(mj['best_meteor']['report']['ang_vel'])[0:5])
      template = template.replace("{ANG_SEP}", str(mj['best_meteor']['report']['ang_dist'])[0:5])

      if "cp" in mj['best_meteor']:
         cp = mj['best_meteor']['cp']
         mj['cp'] = cp
         del(mj['best_meteor']['cp'])

      if "cp" in mj:
         cp = mj['cp']

         print(cp)
         template = template.replace("{RA}", str(cp['ra_center'])[0:5])
         template = template.replace("{DEC}", str(cp['dec_center'])[0:5])
         template = template.replace("{AZ}", str(cp['center_az'])[0:5])
         template = template.replace("{EL}", str(cp['center_el'])[0:5])
         template = template.replace("{POSITION_ANGLE}", str(cp['position_angle'])[0:5])
         template = template.replace("{PIXSCALE}", str(cp['pixscale'])[0:5])
         template = template.replace("{IMG_STARS}", str(len(cp['user_stars'])))
         if "cat_image_stars" in cp:
            template = template.replace("{CAT_STARS}", str(len(cp['cat_image_stars'])))
         else:
            template = template.replace("{CAT_STARS}", "")
         if "total_res_px" in cp:
            template = template.replace("{RES_PX}", str(cp['total_res_px'])[0:5])
            template = template.replace("{RES_DEG}", str(cp['total_res_deg'])[0:5])
         else:
            template = template.replace("{RES_PX}", "")
            template = template.replace("{RES_DEG}", "")
            cp['total_res_px'] = 99
            cp['total_res_deg'] = 99

   #if "total_res_px" not in cp:
   #   cp['total_res_px'] = 99
   #   cp['total_res_deg'] = 99

   if cfe("/mnt/ams2" + CACHE_VDIR, 1) == 0:
      if "mp4" in meteor_file:
         vid = meteor_file.replace(".json", ".mp4")
      else:
         vid = meteor_file.replace(".json", ".mp4")
      cmd = "./Process.py roi_mfd " + METEOR_DIR + vid
      print(cmd)
      os.system(cmd)
   print("CACHE:", CACHE_VDIR) 

   if cfe(mjrf) == 1:
      mjr = load_json_file(mjrf)
      if "total_res_px" not in mjr['cal_params']:
         mjr['cal_params']['total_res_px'] = 99
         mjr['cal_params']['total_res_deg'] = 99
         mjr['cal_params']['cat_image_stars'] = []

      if np.isnan(mjr['cal_params']['total_res_px']) or mjr['cal_params']['total_res_px'] is None or len(mjr['cal_params']['cat_image_stars']) == 0:
         mjr['cal_params']['total_res_px'] = 9999
         mjr['cal_params']['total_res_deg'] = 9999


      frame_table_rows = frames_table(mjr, base_name, CACHE_VDIR)
      cal_params_js_var = "var cal_params = " + str(mjr['cal_params'])
      mfd_js_var = "var meteor_frame_data = " + str(mjr['meteor_frame_data'])
      crop_box_js_var = "var crop_box = " + str(mjr['crop_box'])
   else:
      cal_params_js_var = ""
      mfd_js_var = ""
      crop_box = ""
      crop_box_js_var = ""
      frame_table_rows = ""

   lc_html = light_curve_url(METEOR_DIR + sd_trim , mj)
  
   fn, vdir = fn_dir(meteor_file)
   div_id = fn.replace(".mp4", "")
   jsid = div_id.replace("_", "")

   template = template.replace("{JSID}", jsid)
   template = template.replace("{CROP_BOX}", crop_box_js_var)
   template = template.replace("{CAL_PARAMS}", cal_params_js_var)
   template = template.replace("{METEOR_FRAME_DATA}", mfd_js_var)
   template = template.replace("{FRAME_TABLE_ROWS}", frame_table_rows)
   template = template.replace("{STAR_ROWS}", "")
   template = template.replace("{LIGHTCURVE_URL}", lc_html)
   return(template)   



def frames_table(mjr, base_name, CACHE_VDIR):

   if True:
      # check for reduced data
      #dt, fn, x, y, w, h, oint, ra, dec, az, el
      #frames_table = "<table border=1><tr><td></td><td>Time</td><td>Frame</td><td>X</td><td>Y</td><td>W</td><td>H</td><td>Int</td><td>Ra</td><td>Dec</td><td>Az</td><td>El</td></tr>"
      frames_table = "\n"
      for mfd in mjr['meteor_frame_data']:
         dt, fn, x, y, w, h, oint, ra, dec, az, el = mfd
         date, dtime = dt.split(" ")
         fnid = "{:04d}".format(mfd[1])
         frame_url = CACHE_VDIR + base_name + "-frm" + fnid + ".jpg?r=" + str(time.time())
         frames_table += """<tr id='fr_{:d}' data-org-x='{:d}' data-org-y='{:d}'>""".format(mfd[1], mfd[2], mfd[3])
         frames_table += """<td><div class="st" hidden style="Background-color:#ff0000"></div></td>"""
         img_id = "img_" + str(mfd[1])
         frames_table += """<td><img id='""" + img_id + """' alt="Thumb #'""" + str(mfd[1]) + """'" src='""" +frame_url+ """' width="50" height="50" class="img-fluid smi select_meteor" style="border-color:#ff0000"></td>"""

         frames_table += """<td>{:d}</td><td>{:s} </td>""".format(int(fn), str(dtime))
         frames_table += "<td> {:0.2f} / {:0.2f}</td>".format(ra, dec)
         frames_table += "<td>{:s} / {:s}</td>".format(str(az)[0:5],str(el)[0:5])
         frames_table += """<td>{:s} / {:s}</td><td>{:s} / {:s}</td><td>{:s}</td>""".format(str(x), str(y), str(w), str(h), str(int(oint)))
         frames_table += """<td><a class="btn btn-danger btn-sm delete_frame"><i class="icon-delete"></i></a></td>"""
         frames_table += """<td class="position-relative"><a class="btn btn-success btn-sm select_meteor"><i class="icon-target"></i></a></td>"""
         frames_table += "<td></td><td></td><td></td></tr>\n"

        #table_tbody_html+= '<tr id="fr_'+frame_id+'" data-org-x="'+v[2]+'" data-org-y="'+v[3]+'">

        #<td><div class="st" hidden style="background-color:'+all_colors[i]+'"></div></td>'
        #<td><img alt="Thumb #'+frame_id+'" src='+thumb_path+'?c='+Math.random()+' width="50" height="50" class="img-fluid smi select_meteor" style="border-color:'+all_colors[i]+'"/></td>

        #table_tbody_html+=
        #table_tbody_html+= '<td>'+frame_id+'</td><td>'+_time[1]+'</td><td>'+v[7]+'&deg;/'+v[8]+'&deg;</td><td>'+v[9]+'&deg;/'+v[10]+'&deg;</td><td>'+ parseFloat(v[2])+'/'+parseFloat(v[3]) +'</td><td>'+ v[4]+'x'+v[5]+'</td>';
        #table_tbody_html+= '<td>'+v[6]+'</td>';

   return(frames_table)   

def light_curve_url(sd_video_file, mj):
   light_curve_file = sd_video_file.replace('.mp4','-lightcurve.jpg')
   if(cfe(light_curve_file) == 1):
      lc_url = '<a class="d-block nop text-center img-link-n" href="'+light_curve_file+'"><img  src="'+light_curve_file+'" class="mt-2 img-fluid"></a>'
   else:
      if "best_meteor" in mj:
         lc_url = graph_light_curve(mj)
      else:
         lc_url = ""
         #lc_url = "<div class='alert error mt-4'><iframe scolling=no src=" + light_curve_url  + " width=100% height=640></iframe></div>"
   return(lc_url)

def graph_light_curve(mj):
   x1_vals = ""
   y1_vals = ""
   for i in range(0, len(mj['best_meteor']['ofns'])):
      if x1_vals != "":
         x1_vals += ","
         y1_vals += ","
      x1_vals += str(mj['best_meteor']['ofns'][i])
      y1_vals += str(mj['best_meteor']['oint'][i])
   gurl = "/dist/plot.html?"
   gurl += "title=Meteor_Light_Curve&xat=Intensity&yat=Frame&x1_vals=" + x1_vals + "&y1_vals=" + y1_vals
   return(gurl)
