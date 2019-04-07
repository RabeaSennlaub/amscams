#!/usr/bin/python3
from datetime import datetime
import datetime as ddt
import pymap3d as pm

from sympy import Point3D, Line3D, Segment3D, Plane
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib.UtilLib import convert_filename_to_date_cam, haversine, calc_radiant
from lib.FileIO import cfe
import simplekml

from mpl_toolkits import mplot3d
import math
from lib.FileIO import load_json_file, save_json_file
from lib.WebCalib import HMS2deg

def sync_meteor_frames(meteor):
   max_len = 0
   longest_key = ""
   sync_frames = {}
   observers = []
   for key in meteor['vel_data']:
      leng = len(meteor['vel_data'][key])
      print(key,leng) 
      if leng > max_len:
         longest_key = key
         max_len = leng
   master_key = longest_key
   sync_frames[master_key] = {}


   #create parent object
   for key in meteor['vel_data']:
      start_time = meteor['vel_data'][key][0][0]
      dt_start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
      observers.append(key)
      fld_ft = key + "_ft" 
      fld_fn = key + "_fn" 
      fld_lon = key + "_lon" 
      fld_lat = key + "_lat" 
      fld_alt = key + "_alt" 
      fld_dfs = key + "_dfs" 
      fld_vfs = key + "_vfs" 
      if key != master_key:
         print("Child observation")
      else:
         #for ec in range(0,25):
         #   sync_frames[master_key][ec] = {}
         #   time_diff = (25-ec)/25
         #   dt_child_time = dt_start_time + ddt.timedelta(0,time_diff)
         #   st_child_time = dt_child_time.strftime('%Y-%m-%d %H:%M:%S.%f')
         #   sync_frames[master_key][ec][fld_ft] = st_child_time

         ec = 0
         for vel_data in meteor['vel_data'][key]:
            ft,fn,lon,lat,alt,dist_from_last_point,vel_from_last_point,dist_from_start,vel_from_start = vel_data
            sync_frames[master_key][ec] = {}
            sync_frames[master_key][ec][fld_ft] = ft
            sync_frames[master_key][ec][fld_fn] = fn
            sync_frames[master_key][ec][fld_lon] = lon
            sync_frames[master_key][ec][fld_lat] = lat
            sync_frames[master_key][ec][fld_alt] = alt
            sync_frames[master_key][ec][fld_dfs] = dist_from_start 
            sync_frames[master_key][ec][fld_vfs] = vel_from_start 
            ec = ec + 1
         for ec in range(ec,ec+25):
            sync_frames[master_key][ec] = {}

   #Add children to parent object
   for key in sync_frames:
      print("MIKE:",key)

   time_diffs = []

   #SYNC FRAMES BY DISTANCE
   for key in meteor['vel_data']:
      ms_fld_ft = master_key + "_ft" 
      fld_ft = key + "_ft" 
      fld_fn = key + "_fn" 
      fld_lon = key + "_lon" 
      fld_lat = key + "_lat" 
      fld_alt = key + "_alt" 
      fld_dfs = key + "_dfs" 
      fld_vfs = key + "_vfs" 
      if key == master_key:
         print("Parent observation/master key:", master_key)
      else:
         ec = 0
         for vel_data in meteor['vel_data'][key]:
            ft,fn,lon,lat,alt,dist_from_last_point,vel_from_last_point,dist_from_start,vel_from_start = vel_data
            print("Child observation")

            best_parent_frame,least_dist = find_best_frame_by_dist(sync_frames, master_key, key, lon,lat,alt)


            parent_time = sync_frames[master_key][best_parent_frame][ms_fld_ft] 

            dt_parent_time = datetime.strptime(parent_time, "%Y-%m-%d %H:%M:%S.%f")
            dt_child_time = datetime.strptime(ft, "%Y-%m-%d %H:%M:%S.%f")

            tdiff = (dt_parent_time - dt_child_time).total_seconds()
            time_diffs.append(tdiff)

            print("DIST MATCH CHILD:", ft,fn, "PARENT:", best_parent_frame, parent_time, least_dist, tdiff)
            ec = ec + 1
   
   time_diff = float(np.median(time_diffs))
   print("TIME SHIFT DIFF:", time_diff)

   #ADJUST TIME AND SYNC BY TIME
   for key in meteor['vel_data']:
      ms_fld_ft = master_key + "_ft" 
      ms_fld_lat = master_key + "_lat" 
      ms_fld_lon = master_key + "_lon" 
      ms_fld_alt = master_key + "_alt" 
      fld_ft = key + "_ft" 
      if key == master_key:
         print("Parent observation/master key:", master_key)
      else:
         ec = 0
         fld_ft = key + "_ft" 
         fld_fn = key + "_fn" 
         fld_lon = key + "_lon" 
         fld_lat = key + "_lat" 
         fld_alt = key + "_alt" 
         fld_dfs = key + "_dfs" 
         fld_vfs = key + "_vfs" 
    
         for vel_data in meteor['vel_data'][key]:
            ft,fn,lon,lat,alt,dist_from_last_point,vel_from_last_point,dist_from_start,vel_from_start = vel_data
            print("Child observation")

            dt_child_time = datetime.strptime(ft, "%Y-%m-%d %H:%M:%S.%f")
            dt_child_time = dt_child_time + ddt.timedelta(0,time_diff)

            best_parent_frame,least_time = find_best_frame_by_time(sync_frames, master_key, key, dt_child_time)

            if ms_fld_ft in sync_frames[master_key][best_parent_frame]:
               parent_time = sync_frames[master_key][best_parent_frame][ms_fld_ft] 

               dt_parent_time = datetime.strptime(parent_time, "%Y-%m-%d %H:%M:%S.%f")
               tdiff = abs((dt_parent_time - dt_child_time).total_seconds())
               ec = best_parent_frame 
               sync_frames[master_key][ec][fld_ft] = ft
               sync_frames[master_key][ec][fld_fn] = fn

            if ms_fld_lat in sync_frames[master_key][best_parent_frame]:
               parent_lat = sync_frames[master_key][best_parent_frame][ms_fld_lat] 
               parent_lon = sync_frames[master_key][best_parent_frame][ms_fld_lon] 
               parent_alt = sync_frames[master_key][best_parent_frame][ms_fld_alt] 
               best_parent_frame_dist,least_dist = find_best_frame_by_dist(sync_frames, master_key, key, lon,lat,alt)
               time_adj_dist = calc_point_dist([parent_lon,parent_lat,parent_alt],[lon,lat,alt])

               sync_frames[master_key][ec][fld_lon] = lon
               sync_frames[master_key][ec][fld_lat] = lat
               sync_frames[master_key][ec][fld_alt] = alt
               sync_frames[master_key][ec][fld_dfs] = dist_from_start 
               sync_frames[master_key][ec][fld_vfs] = vel_from_start 


   rpt = ""
   for key in sync_frames:
      for ekey in sync_frames[master_key]:
         #print(key, ekey)
         rpt = str(key) + " " + str(ekey)
         for obs in observers:
            fl_ft = obs + "_ft"
            fl_fn = obs + "_fn"
            fl_lon = obs + "_lon"
            fl_lat = obs + "_lat"
            fl_alt = obs + "_alt"
            if fl_ft in sync_frames[master_key][ekey]:
               rpt = rpt + " " + str(sync_frames[master_key][ekey][fl_ft]) + " " 
            if fl_lon in sync_frames[master_key][ekey]:
               rpt = rpt + str(sync_frames[master_key][ekey][fl_fn]) + " " 
               rpt = rpt + str(sync_frames[master_key][ekey][fl_lon]) + " " 
               rpt = rpt + str(sync_frames[master_key][ekey][fl_lat]) + " " 
               rpt = rpt + str(sync_frames[master_key][ekey][fl_alt])  + " " 
            #if len(rpt) > 5:
         print(rpt)
   meteor['sync_frames'] = sync_frames
   make_sync_kml(sync_frames,meteor,master_key,observers)
   return(meteor)

def make_sync_kml(sync_frames,meteor,master_key,obs_sol):

   curve_data = {}
   for key in obs_sol:
      curve_data[key] = {}
      curve_data[key]['xs'] = []
      curve_data[key]['ys'] = []


   kml_file = meteor_file.replace(".json", ".kml")
   kml = simplekml.Kml()

   observers = {}
   done = {}
   obs_folder = kml.newfolder(name='Stations')
   for key in meteor:
      if "obs" in key:
         obs_lon = meteor[key]['x2_lat']
         obs_lat = meteor[key]['y2_lon']
         observers[key] = {}
         observers[key]['lat'] = obs_lat
         observers[key]['lon'] = obs_lon
         if "key" not in done:
            point = obs_folder.newpoint(name=key,coords=[(obs_lon,obs_lat)])
         done[key] = 1

   colors = [
      'FF641E16',
      'FF512E5F',
      'FF154360',
      'FF0E6251',
      'FF145A32',
      'FF7D6608',
      'FF78281F',
      'FF4A235A',
      'FF1B4F72',
      'FF0B5345',
      'FF186A3B',
      'FF7E5109'
   ]
   meteor['final_solution']
   start_x,start_y,start_z = meteor['final_solution']['meteor_start_point']
   end_x,end_y,end_z = meteor['final_solution']['meteor_end_point']
   poly = kml.newpolygon(name='Meteor Track')
   poly.outerboundaryis = [(start_x,start_y,start_z*1000),(end_x,end_y,end_z*1000),(end_x,end_y,0),(start_x,start_y,0)]
   poly.altitudemode = simplekml.AltitudeMode.relativetoground


   cc = 0
   for key in sync_frames:
      for ekey in sync_frames[master_key]:
         for obs in obs_sol:
            xob,mob = obs.split("-") 
            olon = observers[mob]['lon']
            olat = observers[mob]['lat']
            #sol_key = obs 
            #sol_folder = kml.newfolder(name=sol_key)
            fl_ft = obs + "_ft"
            fl_fn = obs + "_fn"
            fl_lon = obs + "_lon"
            fl_lat = obs + "_lat"
            fl_alt = obs + "_alt"
            fl_vfs = obs + "_vfs"
            #if fl_ft in sync_frames[master_key][ekey]:
            #   rpt = rpt + " " + str(sync_frames[master_key][ekey][fl_ft]) + " " 
            if fl_lon in sync_frames[master_key][ekey]:
               ft = sync_frames[master_key][ekey][fl_ft]
               lon = float(sync_frames[master_key][ekey][fl_lon])  
               lat = float(sync_frames[master_key][ekey][fl_lat])  
               alt = float(sync_frames[master_key][ekey][fl_alt])  * 1000
               vfs = float(sync_frames[master_key][ekey][fl_vfs])  
               color = colors[cc]
               #point = sol_folder.newpoint(coords=[(lon,lat,alt)])
               point = kml.newpoint(coords=[(lon,lat,alt)])

               point.altitudemode = simplekml.AltitudeMode.relativetoground
               point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

               line = kml.newlinestring(name="", description="", coords=[(olon,olat,0),(lon,lat,alt)])
               line.altitudemode = simplekml.AltitudeMode.relativetoground
               line.linestyle.color = color

               point.style.iconstyle.color = color
               line = kml.newlinestring(name="", description="", coords=[(lon,lat,0),(lon,lat,alt)])
               line.altitudemode = simplekml.AltitudeMode.relativetoground
               line.linestyle.color = color
               fts = float(str(ft)[-5:-1])
               curve_data[obs]['xs'].append(fts)
               curve_data[obs]['ys'].append(vfs)
         cc = cc + 1
         if cc > len(colors)-1:
            cc = 0

   oc = 0
   pltz = {}
   for obs in curve_data:
      pltz[obs], = plt.plot(curve_data[obs]['xs'], curve_data[obs]['ys'])
      oc = oc + 1

   code1 = ""
   code2 = ""
   for obs in pltz:
      print(obs)
      if code1 != "":
         code1 = code1 + ","
      if code2 != "":
         code2 = code2 + ","
      code1 = code1 + "pltz['" + obs + "']" 
      code2 = code2 + "\"" + obs + "\""
   fcode = "plt.legend((" + code1 + "),(" + code2 + "))"
   print(fcode)
   eval(fcode)
      #plt.legend(pltz[obs], obs)

   #plt.show()
   plt.title('Meteor Velocity in KM/Second')
   plt.ylabel('KM/Sec From Start')
   plt.xlabel('Frame 1/25 sec')
   vel_fig_file = meteor['meteor_file'].replace(".json", "-fig_vel.png")
   plt.savefig(vel_fig_file)
   print(vel_fig_file)

   kml_file = vel_fig_file.replace("-fig_vel.png",".kml")
   kml.save(kml_file)
   print(kml_file)

def calc_point_dist(p1,p2):
      p1_lon = p1[0]
      p1_lat = p1[1]
      p1_alt = p1[2]
      p2_lon = p2[0]
      p2_lat = p2[1]
      p2_alt = p2[2]

      x1, y1, z1 = pm.geodetic2ecef(p1_lat,p1_lon,p1_alt)
      x2, y2, z2 = pm.geodetic2ecef(p2_lat,p2_lon,p2_alt)

      dt = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
      #print("DISTANCE:", dt)
      return(dt)


def find_best_frame_by_time(sync_frames,master_key,child_key,dt_child_time):
   least_time = 100
   best_match = 0
   #dt_child_time = datetime.strptime(child_time, "%Y-%m-%d %H:%M:%S.%f")
   for key in sync_frames:
      for ekey in sync_frames[master_key]:
         fld_ft = key + "_ft" 
         if fld_ft in sync_frames[master_key][ekey]:
            ptime = sync_frames[master_key][ekey][fld_ft] 
            dt_parent_time = datetime.strptime(ptime, "%Y-%m-%d %H:%M:%S.%f")
            tdiff = abs((dt_parent_time - dt_child_time).total_seconds())
            if tdiff < least_time:
               least_time = tdiff
               best_match = ekey
   return(best_match,least_time)

def find_best_frame_by_dist(sync_frames,master_key,child_key,lon,lat,alt):
   least_dist = 100000000
   best_match = 0
   for key in sync_frames:
      for ekey in sync_frames[master_key]:
         fld_lon = key + "_lon" 
         fld_lat = key + "_lat" 
         fld_alt = key + "_alt" 
         #print(master_key,child_key,key,ekey)
         #print(sync_frames[master_key][ekey])
         if fld_lon in sync_frames[master_key][ekey]:
            p_lon = sync_frames[master_key][ekey][fld_lon] 
            p_lat = sync_frames[master_key][ekey][fld_lat] 
            p_alt = sync_frames[master_key][ekey][fld_alt] 
            p1 = [p_lon,p_lat,p_alt]
            p2 = [lon,lat,alt]
            this_dist = calc_point_dist(p1,p2) 
            if this_dist < least_dist:
               least_dist = this_dist
               best_match = ekey

         #print(key, ekey, p_lat, p_lon, p_alt, lat,lon,alt,this_dist)
   return(best_match,least_dist)

def find_closest_frame(master_ftime, time_list ):
   mfc = 0
   lc = 0
   temp = []
   for stime in time_list:
      #xtime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S.%f")
      xtime = stime
      tdiff = abs((master_ftime - xtime).total_seconds())
      temp.append((master_ftime, xtime, lc, mfc, tdiff))
      lc = lc + 1
   temp_sorted = sorted(temp, key=lambda x: x[4], reverse=False)
   #for mft, tft, tfc, mfc, tdiff in temp_sorted:
   match = temp_sorted[0]
   print("BEST MATCHED FRAME:", match[1])
   return(match[1],match[4])


def velocity_curve(meteor):

   frame_map = {}

   longest_key = ""
   max_len = 0 
   mapped_frame_data = {}
   time_list = [] 
   for key in meteor['vel_data']:
      leng = len(meteor['vel_data'][key])
      print(key,leng) 
      if leng > max_len:
         longest_key = key
         max_len = leng

   for vel in meteor['vel_data'][longest_key]:
      (ft,fn,lon,lat,alt,point_dist,vel,dist_from_start,vel_from_start) = vel
      ft = datetime.strptime(ft, "%Y-%m-%d %H:%M:%S.%f")
      mapped_frame_data[ft] = {}
      if longest_key not in mapped_frame_data[ft]:
         mapped_frame_data[ft][longest_key] = [ft,fn,lon,lat,alt]
         time_list.append(ft)
      print(ft)

   # add 25 frames before and after:
   #st = datetime.strptime(time_list[0], "%Y-%m-%d %H:%M:%S.%f")
   #et = datetime.strptime(time_list[-1], "%Y-%m-%d %H:%M:%S.%f")
   st = time_list[0]
   et = time_list[-1]

   for i in range(1,100):
      extra_sec = i/25 
      frame_time = st - ddt.timedelta(seconds=extra_sec)
      mapped_frame_data[frame_time] = {}
      time_list.append(frame_time)
      frame_time = et + ddt.timedelta(seconds=extra_sec)
      mapped_frame_data[frame_time] = {}
      time_list.append(frame_time)
 
     
   time_list = sorted(time_list)
 

   for key in meteor['vel_data']:
      print("MAP TIME/FRAMES of :", key)
      #if key not in mapped_frame_data[ft]:
      if True:
         for vel in meteor['vel_data'][key]:
            (ft,fn,lon,lat,alt,point_dist,vel,dist_from_start,vel_from_start) = vel
            this_ft = datetime.strptime(ft, "%Y-%m-%d %H:%M:%S.%f")
            bftm,tdiff  = find_closest_frame(this_ft, time_list)
            bftms = bftm.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            mapped_frame_data[bftm][key] = [ft,fn,lon,lat,alt]


   print(mapped_frame_data)

   print("MOST FRAMES IN SOLUTION:", longest_key, max_len) 
   exit()

   obs1_tp = len(meteor['vel_curve_x']['obs2-obs1'])
   obs2_tp = len(meteor['vel_curve_x']['obs1-obs2'])
   if obs1_tp < obs2_tp:
      master_time = meteor['vel_curve_x']['obs2-obs1']
      slave_time = meteor['vel_curve_x']['obs1-obs2']
   else:
      master_time = meteor['vel_curve_x']['obs1-obs2']
      slave_time = meteor['vel_curve_x']['obs2-obs1']

   stc = 0
   for stime in slave_time:
      frame_map[stc] = {} 
      frame_map[stc]['sobs_time'] = stime
      frame_map[stc]['mobs_time'] = ""
      frame_map[stc]['mobs_frame'] = ""
      frame_map[stc]['mobs_tdiff'] = ""
      stc = stc + 1

   mfc = 0

   print("MASTER TIME: ", len(master_time))
   print("SLAVE TIME: ", len(slave_time))
   for ftime in master_time:
      master_ftime = datetime.strptime(ftime, "%Y-%m-%d %H:%M:%S.%f")
      best_frame_match,tdiff  = find_closest_frame(master_ftime, slave_time, mfc)
      frame_map[best_frame_match]['mobs_time'] = master_ftime
      frame_map[best_frame_match]['mobs_frame'] = mfc
      frame_map[best_frame_match]['mobs_tdiff'] = tdiff
      
      mfc = mfc + 1

   for frame_num in frame_map:
      sobs_time = frame_map[frame_num]['sobs_time']
      mobs_time = frame_map[frame_num]['mobs_time']
      mobs_frame = frame_map[frame_num]['mobs_frame']
      mobs_tdiff = frame_map[frame_num]['mobs_tdiff']
      print(frame_num, sobs_time, mobs_frame, mobs_time, mobs_tdiff)

   return(meteor)


def compute_velocity(meteor):
   meteor['vel_data'] = {}
   meteor['vel_curve_x'] = {}
   meteor['vel_curve_y'] = {}
   crc = 0
   for key in meteor['meteor_points_lat_lon']:
      points = []
      pc = 0
      vel_data = []
      curve_x = []
      curve_y = []

      for ft,fn,lon,lat,alt in meteor['meteor_points_lat_lon'][key]:
        
           
         point = Point3D(lon,lat,alt) 
         points.append(point)
         point_dist = 0
         vel = 0
         dist_from_start = 0
         vel_from_start = 0
         if pc > 0:
            # compute distance between last point and this point
            fd = fn - vel_data[pc-1][1]
            tf = fn - vel_data[0][1]
            point_dist = points[pc].distance(points[pc-1]).evalf()
            dist_from_start = points[pc].distance(points[0]).evalf()
            vel = point_dist / (fd * .04)
            vel_from_start = dist_from_start / (tf * .04)
            print("POINT DIST:", point_dist, "KM ", vel, "KM per second", fn, vel_data[pc-1][1],fd)
            curve_x.append(str(ft))
            curve_y.append(float(vel))
         vel_data.append((ft,fn,lon,lat,alt,float(point_dist),float(vel),float(dist_from_start),float(vel_from_start)))
         pc = pc + 1

      meteor['vel_data'][key] = vel_data
      meteor['vel_curve_x'][key] = curve_x
      meteor['vel_curve_y'][key] = curve_y

   print(vel_data)
   meteor['final_solution'] = {}
   #meteor['final_solution']['vel_avg'] = float(vel_from_start)
   #meteor

   obs1_tp = len(meteor['vel_curve_x']['obs2-obs1'])
   obs2_tp = len(meteor['vel_curve_x']['obs1-obs2'])
   if obs1_tp > obs2_tp:
      master_time = meteor['vel_curve_x']['obs2-obs1']
   else:
      master_time = meteor['vel_curve_x']['obs1-obs2']

   for time in master_time:
      print(time)

   return(meteor)
   #line = Line3D(Point3D(mod['ObsX'],mod['ObsY'],mod['ObsZ']),Point3D(veX,veY,veZ))

def lin(z, c_xz, c_yz,m_xz,m_yz):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y

def fit_points_to_line(meteor,meteor_json_file):
   xs = []
   ys = []
   zs = []
   sxs = []
   sys = []
   szs = []
   exs = []
   eys = []
   ezs = []
   for key in meteor['vel_data']:
      print (meteor['vel_data'][key])
      sxs.append(meteor['vel_data'][key][0][2])
      sys.append(meteor['vel_data'][key][0][3])
      szs.append(meteor['vel_data'][key][0][4])
      exs.append(meteor['vel_data'][key][-1][2])
      eys.append(meteor['vel_data'][key][-1][3])
      ezs.append(meteor['vel_data'][key][-1][4])
      if key != 'obs1-obs3' and key != 'obs3-obs1':
         for ft,fn,lon,lat,alt,dist,vel,tdist,tvel in meteor['vel_data'][key]:
            xs.append(lon)
            ys.append(lat)
            zs.append(alt)

   A_xz = np.vstack((xs,np.ones(len(xs)))).T
   m_xz, c_xz = np.linalg.lstsq(A_xz, zs,rcond=None)[0]

   A_yz = np.vstack((ys, np.ones(len(ys)))).T
   m_yz, c_yz = np.linalg.lstsq(A_yz, zs,rcond=None)[0]

   fig = plt.figure()
   ax = Axes3D(fig)
   zz = np.linspace(0,100)
   xx,yy = lin(zz,c_xz, c_yz,m_xz,m_yz)
   ax.scatter(xs, ys, zs)
   ax.plot(xx,yy,zz,c='r')
   plt.savefig('test.png')
   meteor['final_solution']['0km'] = [xx[0],yy[0],zz[0]]
   meteor['final_solution']['100km'] = [xx[-1],yy[-1],zz[-1]]
   start_alt = np.mean(szs)
   end_alt = np.mean(ezs)

   se_zz = [start_alt,end_alt] 
   se_xx,se_yy = lin(se_zz,c_xz, c_yz,m_xz,m_yz)

   print(se_xx)
   print(se_yy)
   print(se_zz)

   meteor['final_solution']['meteor_start_point'] = [se_xx[0],se_yy[0],se_zz[0]]
   meteor['final_solution']['meteor_end_point'] = [se_xx[1],se_yy[1],se_zz[1]]
   print("0KM:", meteor['final_solution']['0km'])
   print("100KM:", meteor['final_solution']['100km'])
   print("Start:", meteor['final_solution']['meteor_start_point'])
   print("End:", meteor['final_solution']['meteor_end_point'])

   track_length, bearing = haversine(meteor['final_solution']['0km'][0],meteor['final_solution']['0km'][1],meteor['final_solution']['100km'][0],meteor['final_solution']['100km'][1])
   hd_datetime, hd_cam, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(meteor_json_file)

   arg_date = hd_y + "-" + hd_m + "-" + hd_d
   arg_time = hd_h + ":" + hd_M + ":" + hd_s

   #def calc_radiant(end_lon, end_lat, end_alt, start_lon, start_lat, start_alt, arg_date, arg_time):
   rah,dech,az,el,track_dist,entry_angle = calc_radiant(meteor['final_solution']['0km'][1],meteor['final_solution']['0km'][0],0,meteor['final_solution']['100km'][1],meteor['final_solution']['100km'][0], 100,arg_date, arg_time)
   rah = str(rah).replace(":", " ")
   dech = str(dech).replace(":", " ")
   ra,dec= HMS2deg(str(rah),str(dech))

   
   meteor['final_solution']['radiant_az'] = az
   meteor['final_solution']['radiant_el'] = el
   meteor['final_solution']['radiant_ra'] = ra
   meteor['final_solution']['radiant_dec'] = dec

   print("RADIANT:", ra,dec, az,el)
   return(meteor)

def make_kmz(meteor):
   kml_file = meteor_file.replace(".json", ".kml")
   kml = simplekml.Kml()

   observers = {}
   done = {}
   obs_folder = kml.newfolder(name='Stations')
   for key in meteor:
      if "obs" in key:
         obs_lon = meteor[key]['x2_lat'] 
         obs_lat = meteor[key]['y2_lon'] 
         observers[key] = {}
         observers[key]['lat'] = obs_lat
         observers[key]['lon'] = obs_lon
         if "key" not in done:
            point = obs_folder.newpoint(name=key,coords=[(obs_lon,obs_lat)])
         done[key] = 1
         
         colors = [
            'FF641E16',
            'FF512E5F',
            'FF154360',
            'FF0E6251',
            'FF145A32',
            'FF7D6608',
            'FF78281F',
            'FF4A235A',
            'FF1B4F72',
            'FF0B5345',
            'FF186A3B',
            'FF7E5109'
         ]

   for key in meteor['meteor_points_lat_lon']:
      cc = 0

      sol_key = key
      sol_folder = kml.newfolder(name=sol_key)

      for ft,fn,lon,lat,alt in meteor['meteor_points_lat_lon'][key]:
         if cc > len(colors) -1:
            cc = 0 
         obs1, obs2 = key.split("-")
         olat = observers[obs2]['lat']
         olon = observers[obs2]['lon']
         print("KEY: ", key, olat, olon)
         alt = alt * 1000
         #print(lat,lon,alt)
         point = sol_folder.newpoint(coords=[(lon,lat,alt)])
         point.altitudemode = simplekml.AltitudeMode.relativetoground    
         #point.style.iconstyle.icon.href = None
         point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
         color = colors[cc]
         print(color)
         point.style.iconstyle.color = color
         line = sol_folder.newlinestring(name="", description="", coords=[(olon,olat,0),(lon,lat,alt)])
         line.altitudemode = simplekml.AltitudeMode.relativetoground
         line.linestyle.color = color
         cc = cc + 1

   #zkmx, zkmy, zkmz = meteor['final_solution']['0km']
   #okmx,okmy,okmz = meteor['final_solution']['100km']

   start_x,start_y,start_z = meteor['final_solution']['meteor_start_point']
   end_x,end_y,end_z = meteor['final_solution']['meteor_end_point']
   poly = kml.newpolygon(name='Meteor Track')
   poly.outerboundaryis = [(start_x,start_y,start_z*1000),(end_x,end_y,end_z*1000),(end_x,end_y,0),(start_x,start_y,0)]
   poly.altitudemode = simplekml.AltitudeMode.relativetoground    

   print(kml_file)
   kml.save(kml_file)

def plot_meteor_ms(meteor):
   fig_file = meteor_file.replace(".json", "-fig2.png")
   fig = plt.figure()
   ax = Axes3D(fig)
   xs = []
   ys = []
   zs = []
  
   for key in meteor:
      if "obs" in key:

      #nmx = (mx / (111.32 * math.cos(meteor['obs1']['ObsY']*math.pi/180))) + meteor['obs1']['lat']
      #nmy = (my / (111.14)) + meteor['obs1']['lon']

         #ox = (float(meteor[key]['ObsX'])/ (111.32 * math.cos(meteor['obs1']['ObsY']*math.pi/180))) + meteor['obs1']['lat']
         #oy = (float(meteor[key]['ObsY']) / 111.14) + meteor['obs1']['lon']

         ox = meteor['obs1']['lon'] + meteor[key]['ObsX'] / (111.32 * math.cos(meteor['obs1']['lat']*math.pi/180))
         oy = meteor['obs1']['lat'] + (meteor[key]['ObsY'] / 111.14) 

         xs.append(ox)
         ys.append(oy)
         zs.append(float(meteor[key]['ObsZ']))
         ax.text(ox, oy,meteor[key]['ObsZ'],meteor[key]['station_name'],fontsize=10)
         meteor[key]['x2_lat'] = ox
         meteor[key]['y2_lon'] = oy
       

   #ax.text(Obs1Lon, Obs1Lat,Obs1Alt,Obs1Station,fontsize=10)
   ax.scatter3D(xs,ys,zs,c='r',marker='o')

   xs = []
   ys = []
   zs = []
   plane_colors = {}
   plane_colors['obs1'] = 'b'
   plane_colors['obs2'] = 'g'
   plane_colors['obs3'] = 'y'
   cc = 0
   meteor_points_lat_lon = {}
   #long = long1 + BolideX / (111.32 * math.cos(lat1*math.pi/180));
   #lat = lat1 + (BolideY / 111.14);
   for key in meteor['meteor_points']:
      plane_key,line_key = key.split("-")
      meteor_points_lat_lon[key] = []
      ox = meteor['obs1']['lon'] + meteor[line_key]['ObsX'] / (111.32 * math.cos(meteor['obs1']['lat']*math.pi/180))
      oy = meteor['obs1']['lat'] + (meteor[line_key]['ObsY'] / 111.14) 

      oz = meteor[line_key]['ObsZ']
      for ft,fn,x,y,z in meteor['meteor_points'][key]:
         x = meteor['obs1']['lon'] + x / (111.32 * math.cos(meteor['obs1']['lat']*math.pi/180))
         y = (y / 111.14) + meteor['obs1']['lat']


         xs.append(x)
         ys.append(y)
         zs.append(z)
         meteor_points_lat_lon[key].append((ft,fn,x,y,z))
         color = plane_colors[line_key]
         ax.plot([ox,x],[oy,y],[oz,z],c=color) 
   ax.scatter3D(xs,ys,zs,c='r',marker='x')
   meteor['meteor_points_lat_lon'] = meteor_points_lat_lon

   plt.savefig(fig_file)

   #plt.show()
   return(meteor)


def plot_meteor_obs(meteor, meteor_file):
   fig_file = meteor_file.replace(".json", "-fig2.png")
   fig = plt.figure()
   ax = Axes3D(fig)
   #lat = lat1 + (BolideY / 111.14);
   #long = long1 + BolideX / (111.32 * math.cos(lat1*math.pi/180));

   Obs1Station = meteor['obs1']['station_name']
   Obs1Lon = (meteor['obs1']['ObsX'] / (111.32 * math.cos(meteor['obs1']['ObsY']*math.pi/180))) + meteor['obs1']['lat']
   Obs1Lat = (meteor['obs1']['ObsY'] / (111.14)) + meteor['obs1']['lon']
   Obs1Alt = meteor['obs1']['ObsZ'] 
   ax.text(Obs1Lon, Obs1Lat,Obs1Alt,Obs1Station,fontsize=10)

   Obs2Station = meteor['obs2']['station_name']
   Obs2Lon = (meteor['obs2']['ObsX'] / (111.32 * math.cos(meteor['obs2']['ObsY']*math.pi/180))) + meteor['obs2']['lat']
   Obs2Lat = (meteor['obs2']['ObsY'] / (111.14)) + meteor['obs2']['lon']
   Obs2Alt = meteor['obs2']['ObsZ'] 
   ax.text(Obs2Lon, Obs2Lat,Obs2Alt,Obs2Station,fontsize=10)

   print("OBS1:", Obs1Lon, Obs1Lat, Obs1Alt)
   print("OBS2:", Obs2Lon, Obs2Lat, Obs2Alt)

   #x = [meteor['obs1']['ObsX'], meteor['obs2']['ObsX']]
   #y = [meteor['obs1']['ObsY'], meteor['obs2']['ObsY']]
   #z = [meteor['obs1']['ObsZ'], meteor['obs2']['ObsZ']]

   x = [Obs1Lon, Obs2Lon]
   y = [Obs1Lat, Obs2Lat]
   z = [Obs1Alt, Obs2Alt]
   ax.scatter3D(x,y,z,c='r',marker='o')
 
   meteor_points1 = meteor['meteor_points1']
   meteor_points2 = meteor['meteor_points2']

   for mx,my,mz in meteor_points1:
      nmx = (mx / (111.32 * math.cos(meteor['obs1']['ObsY']*math.pi/180))) + meteor['obs1']['lat']
      nmy = (my / (111.14)) + meteor['obs1']['lon']
      ax.plot([Obs1Lon,nmx],[Obs1Lat,nmy],[Obs1Alt,mz],c='g')
   for mx,my,mz in meteor_points1:
      nmx = (mx / (111.32 * math.cos(meteor['obs1']['ObsY']*math.pi/180))) + meteor['obs1']['lat']
      nmy = (my / (111.14)) + meteor['obs1']['lon']
      ax.plot([Obs2Lon,nmx],[Obs2Lat,nmy],[Obs2Alt,mz],c='b')
   ax.set_xlabel('Longitude')
   ax.set_ylabel('Latitude')
   ax.set_zlabel('Altitude')

   plt.savefig(fig_file)

def plot_meteor(meteor, meteor_file):
   fig_file = meteor_file.replace(".json", "-fig1.png")
   fig = plt.figure()
   ax = Axes3D(fig)
   #print(meteor)

   # plot observers
   #ax.scatter3D(x,y,z,c='r',marker='o')

   meteor_points1 = meteor['meteor_points1']
   meteor_points2 = meteor['meteor_points2']

   xs = []
   ys = []
   zs = []
   for mx,my,mz in meteor_points1:
      if mz > 10:
         xs.append(mx)
         ys.append(my)
         zs.append(mz)
   ax.scatter3D(xs,ys,zs,marker='x')

   for mx,my,mz in meteor_points2:
      if mz > 10:
         xs.append(mx)
         ys.append(my)
         zs.append(mz)
   ax.scatter3D(xs,ys,zs,marker='o')

   #plt.show()
   plt.savefig(fig_file)

def compute_ms_solution(meteor):
   vfact = 180
   for key in meteor:
      if "obs" in key:
         vp = []
         for data in meteor[key]['mo_vectors'] :
            ft,fn,vx,vy,vz = data
            veX = meteor[key]['ObsX'] + ( vx * vfact)
            veY = meteor[key]['ObsY'] + ( vy * vfact)
            veZ = meteor[key]['ObsZ'] + ( vz * vfact)
            vp.append((ft,fn,veX,veY,veZ))
         meteor[key]['vector_points'] = vp

   planes = {}
   for key in meteor:
      if "obs" in key :
         mod = meteor[key]
         print(mod) 
         planes[key] = Plane( \
            Point3D(mod['ObsX'],mod['ObsY'],mod['ObsZ']), \
            Point3D(mod['vector_points'][0][2],mod['vector_points'][0][3],mod['vector_points'][0][4]), \
            Point3D(mod['vector_points'][-1][2],mod['vector_points'][-1][3], mod['vector_points'][-1][4]))
   print(planes)

   meteor_points = {}


   for pkey in planes: 
      plane = planes[pkey]
      for key in meteor:
         if "obs" in key and key != pkey:
            mod = meteor[key]
            point_key = pkey + "-" + key
            meteor_points[point_key] = []
            for ft,fn,veX,veY,veZ in mod['vector_points']:

               print("LINE DATA:", mod['ObsX'],mod['ObsY'],mod['ObsZ'],veX,veY,veZ)
               line = Line3D(Point3D(mod['ObsX'],mod['ObsY'],mod['ObsZ']),Point3D(veX,veY,veZ))

               inter = plane.intersection(line)
               print(inter[0])
               mx = float((eval(str(inter[0].x))))
               my = float((eval(str(inter[0].y))))
               mz = float((eval(str(inter[0].z))))
               meteor_points[point_key].append((ft,fn,mx,my,mz))

   meteor['meteor_points'] = meteor_points
   #print(meteor_points)
   return(meteor)      
         

def compute_solution(meteor):
   # vector factor
   vfact = 180 

   # plot line vectors for obs1
   Obs1X = meteor['obs1']['ObsX']
   Obs1Y = meteor['obs1']['ObsY']
   Obs1Z = meteor['obs1']['ObsZ']
   mv = meteor['obs1']['vectors']
   vp1 = []
   for data in mv:
      vx,vy,vz = data
      veX = Obs1X + ( vx * vfact)
      veY = Obs1Y + ( vy * vfact)
      veZ = Obs1Z + ( vz * vfact)
      vp1.append((veX,veY,veZ))
   plane1 = Plane(Point3D(Obs1X,Obs1Y,Obs1Z),Point3D(vp1[0][0],vp1[0][1],vp1[0][2]),Point3D(vp1[-1][0], vp1[-1][1], vp1[-1][2]))

   # plot line vectors for obs2
   Obs2X = meteor['obs2']['ObsX']
   Obs2Y = meteor['obs2']['ObsY']
   Obs2Z = meteor['obs2']['ObsZ']
   mv = meteor['obs2']['vectors']
   vp2 = []
   for data in mv:
      vx,vy,vz = data
      veX = Obs2X + ( vx * vfact)
      veY = Obs2Y + ( vy * vfact)
      veZ = Obs2Z + ( vz * vfact)
      vp2.append((veX,veY,veZ))

 # plot line vectors for obs2
   Obs2X = meteor['obs2']['ObsX']
   Obs2Y = meteor['obs2']['ObsY']
   Obs2Z = meteor['obs2']['ObsZ']
   mv = meteor['obs2']['vectors']
   vp2 = []
   for data in mv:
      vx,vy,vz = data
      veX = Obs2X + ( vx * vfact)
      veY = Obs2Y + ( vy * vfact)
      veZ = Obs2Z + ( vz * vfact)
      vp2.append((veX,veY,veZ))

   plane2 = Plane(Point3D(Obs2X,Obs2Y,Obs2Z),Point3D(vp2[0][0],vp2[0][1],vp2[0][2]),Point3D(vp2[-1][0], vp2[-1][1], vp2[-1][2]))

   meteor_points1 = []
   meteor_points2 = []

   for veX,veY,veZ in vp1:
      line = Line3D(Point3D(Obs1X,Obs1Y,Obs1Z),Point3D(veX,veY,veZ))

      inter = plane2.intersection(line)
      mx = float((eval(str(inter[0].x))))
      my = float((eval(str(inter[0].y))))
      mz = float((eval(str(inter[0].z))))
      meteor_points1.append((mx,my,mz))

   for veX,veY,veZ in vp2:
      line = Line3D(Point3D(Obs2X,Obs2Y,Obs2Z),Point3D(veX,veY,veZ))

      inter = plane1.intersection(line)
      mx = float((eval(str(inter[0].x))))
      my = float((eval(str(inter[0].y))))
      mz = float((eval(str(inter[0].z))))
      meteor_points2.append((mx,my,mz))

   xs = []
   ys = []
   zs = []
   for mx,my,mz in meteor_points1:
      xs.append(mx)
      ys.append(my)
      zs.append(mz)

   for mx,my,mz in meteor_points2:
      xs.append(mx)
      ys.append(my)
      zs.append(mz)

   meteor['meteor_points1'] = meteor_points1
   meteor['meteor_points2'] = meteor_points2
   meteor['vp1'] = vp1 
   meteor['vp2'] = vp2 
   return(meteor)





def plot_xyz(x,y,z,meteor):
   vfact = 180 
   fig = plt.figure()
   ax = Axes3D(fig)
   #line1, line2 = make_lines_for_obs(cart1)
   #x = [line1[0][0],line1[0][1],line2[0][1]]
   #y = [line1[1][0],line1[1][1],line2[1][1]]
   #z = [line1[2][0],line1[2][1],line2[2][1]]


   ax.scatter3D(x,y,z,c='r',marker='o')


   # plot line vectors for obs1 
   Obs1X = meteor['obs1']['ObsX']
   Obs1Y = meteor['obs1']['ObsY']
   Obs1Z = meteor['obs1']['ObsZ']
   mv = meteor['obs1']['vectors']
   vp = []
   for data in mv:
      vx,vy,vz = data
      veX = Obs1X + ( vx * vfact)
      veY = Obs1Y + ( vy * vfact)
      veZ = Obs1Z + ( vz * vfact)
      #ax.plot([Obs1X,veX],[Obs1Y,veY],[Obs1Z,veZ], color='green')
      vp.append((veX,veY,veZ))


   plane1 = Plane(Point3D(Obs1X,Obs1Y,Obs1Z),Point3D(vp[0][0],vp[0][1],vp[0][2]),Point3D(vp[-1][0], vp[-1][1], vp[-1][2]))

   # plot line vectors for obs2
   Obs2X = meteor['obs2']['ObsX']
   Obs2Y = meteor['obs2']['ObsY']
   Obs2Z = meteor['obs2']['ObsZ']
   mv = meteor['obs2']['vectors']
   vp2 = []
   for data in mv:
      vx,vy,vz = data
      veX = Obs2X + ( vx * vfact)
      veY = Obs2Y + ( vy * vfact)
      veZ = Obs2Z + ( vz * vfact)
      vp2.append((veX,veY,veZ))

   plane2 = Plane(Point3D(Obs2X,Obs2Y,Obs2Z),Point3D(vp2[0][0],vp2[0][1],vp2[0][2]),Point3D(vp2[-1][0], vp2[-1][1], vp2[-1][2]))

   meteor_points1 = []
   meteor_points2 = []

   for veX,veY,veZ in vp:
      line = Line3D(Point3D(Obs1X,Obs1Y,Obs1Z),Point3D(veX,veY,veZ))

      inter = plane2.intersection(line)
      mx = float((eval(str(inter[0].x))))
      my = float((eval(str(inter[0].y))))
      mz = float((eval(str(inter[0].z))))
      #ax.scatter3D(mx,my,mz,c='r',marker='x')
      ax.plot([Obs1X,mx],[Obs1Y,my],[Obs1Z,mz],c='g')
      meteor_points1.append((mx,my,mz))


   for veX,veY,veZ in vp2:
      line = Line3D(Point3D(Obs2X,Obs2Y,Obs2Z),Point3D(veX,veY,veZ))

      inter = plane1.intersection(line)
      mx = float((eval(str(inter[0].x))))
      my = float((eval(str(inter[0].y))))
      mz = float((eval(str(inter[0].z))))
      #ax.scatter3D(mx,my,mz,c='r',marker='x')
      ax.plot([Obs2X,mx],[Obs2Y,my],[Obs2Z,mz],c='b')
      meteor_points2.append((mx,my,mz))

   xs = []
   ys = []
   zs = []
   for mx,my,mz in meteor_points1:
      xs.append(mx)
      ys.append(my)
      zs.append(mz)
   ax.scatter3D(xs,ys,zs,marker='x')

   for mx,my,mz in meteor_points2:
      xs.append(mx)
      ys.append(my)
      zs.append(mz)
   ax.scatter3D(xs,ys,zs,marker='o')


   #ax.set_zlim(0, 140)
   #ax.set_xlim(np.min(x)-100, np.max(x)+100)
   #ax.set_ylim(np.min(y)-100, np.max(y)+100)

   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
   #plt.show()

def make_obs_vectors(mo):
   fc = 0
   mo_vectors = []
   for data in mo['meteor_frame_data']:
      frame_time = data[0]
      frame_num = data[1]
      az = data[9]
      el = data[10]
      vx = math.sin(math.radians(az)) * math.cos(math.radians(el))
      vy = math.cos(math.radians(az)) * math.cos(math.radians(el))
      vz = math.sin(math.radians(el))
      mo_vectors.append((frame_time,frame_num,vx,vy,vz))
   return(mo_vectors)

def setup_obs(meteors_obs):

   meteor = {}
   mos = {}
   lats = []
   lons = []
   alts = []
   for i in range(1,len(meteor_obs)+1):
      mokey = 'mo' + str(i)
      obskey = 'obs' + str(i)
      mo = meteor_obs[mokey]
      mos[mokey] = mo
      meteor[obskey] = {}
      lats.append(float(mos[mokey]['cal_params']['site_lat']))
      lons.append(float(mos[mokey]['cal_params']['site_lng']))
      alts.append(float(mos[mokey]['cal_params']['site_alt']) / 1000)

   # determine base lat/lon (use first observer) 


   Obs1Z = alts[0]
   Obs1Y = 0 
   Obs1X = 0 

   meteor['obs1']['station_name'] = mo1['station_name']
   meteor['obs1']['lat'] = lats[0]
   meteor['obs1']['lon'] = lons[0]
   meteor['obs1']['alt'] = alts[0]
   meteor['obs1']['ObsX'] = Obs1X
   meteor['obs1']['ObsY'] = Obs1Y
   meteor['obs1']['ObsZ'] = Obs1Z
   meteor['obs1']['reduction_file'] = mo1['reduction_file']
   meteor['obs1']['mo_vectors'] = make_obs_vectors(meteor_obs['mo1'])


   for i in range(2,len(meteor_obs)+1):
      obskey = "obs" + str(i) 
      mokey = 'mo' + str(i)



      meteor[obskey]['station_name'] = meteor_obs[mokey]['station_name']
      meteor[obskey]['lat'] = lats[i-1]
      meteor[obskey]['lon'] = lons[i-1]
      meteor[obskey]['alt'] = alts[i-1]
      meteor[obskey]['ObsX'] = (lons[i-1] - lons[0])*111.32*math.cos(((lats[0]+lats[i-1])/2)*math.pi/180)
      meteor[obskey]['ObsY'] = (lats[i-1]- lats[0])*111.14
      meteor[obskey]['ObsZ'] = alts[i-1] - alts[0]
      meteor[obskey]['reduction_file'] = meteor_obs[mokey]['reduction_file']
      meteor[obskey]['mo_vectors'] = make_obs_vectors(meteor_obs[mokey])

   return(meteor)



if len(sys.argv) == 3:
   obs1_file, obs2_file = sys.argv[1], sys.argv[2]
   hd_datetime, cam1, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(obs1_file)
   hd_datetime, cam2, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(obs2_file)
   meteor_file = "/mnt/ams2/multi_station/" + hd_y + "_" + hd_m + "_" + hd_d + "/" + hd_y + "_" + hd_m + "_" + hd_d + "_" + hd_h + "_" + hd_M + "_" + hd_s + "_" + cam1 + "_" + cam2 + "-solved.json" 
   mo1 = load_json_file(obs1_file)
   mo2 = load_json_file(obs2_file)
   meteor_obs = {}
   meteor_obs['mo1'] = mo1
   meteor_obs['mo2'] = mo2

   meteor_obs['mo1']['reduction_file'] = obs1_file
   meteor_obs['mo2']['reduction_file'] = obs2_file

if len(sys.argv) == 4:
   obs1_file, obs2_file,obs3_file = sys.argv[1], sys.argv[2], sys.argv[3]
   hd_datetime, cam1, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(obs1_file)
   hd_datetime, cam2, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(obs2_file)
   hd_datetime, cam3, hd_date, hd_y, hd_m, hd_d, hd_h, hd_M, hd_s = convert_filename_to_date_cam(obs3_file)
   meteor_file = "/mnt/ams2/multi_station/" + hd_y + "_" + hd_m + "_" + hd_d + "/" + hd_y + "_" + hd_m + "_" + hd_d + "_" + hd_h + "_" + hd_M + "_" + hd_s + "_" + cam1 + "_" + cam2 + "_" + cam3 + "-solved.json" 
   mo1 = load_json_file(obs1_file)
   mo2 = load_json_file(obs2_file)
   mo3 = load_json_file(obs3_file)
   meteor_obs = {}
   meteor_obs['mo1'] = mo1
   meteor_obs['mo2'] = mo2
   meteor_obs['mo3'] = mo3

   meteor_obs['mo1']['reduction_file'] = obs1_file
   meteor_obs['mo2']['reduction_file'] = obs2_file
   meteor_obs['mo3']['reduction_file'] = obs3_file

if cfe(meteor_file) == 0:
   meteor = setup_obs(meteor_obs)
   meteor = compute_ms_solution(meteor)
   meteor['meteor_file'] = meteor_file
else:
   meteor = load_json_file(meteor_file)
   meteor['meteor_file'] = meteor_file
save_json_file(meteor_file, meteor)


meteor = plot_meteor_ms(meteor)
meteor = compute_velocity(meteor)
meteor = fit_points_to_line(meteor,meteor_file)

meteor  = sync_meteor_frames(meteor)

####meteor  = velocity_curve(meteor)
###save_json_file(meteor_file, meteor)

save_json_file(meteor_file, meteor)
make_kmz(meteor)

