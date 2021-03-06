import sys
import os
import json
import numpy as np 
import statistics 
import requests 
import glob

from lib.FileIO import cfe, save_json_file, load_json_file
from lib.VIDEO_VARS import HD_W, HD_H
from lib.MeteorReduce_Tools import get_cache_path, does_cache_exist
from lib.IIIDLight_Curve import get_json_for_3Dlight_curve


DEFAULT_IFRAME = "<div class='load_if'><iframe width='100%' height='517' style='margin:.5rem auto' frameborder='false' data-src='{CONTENT}'></iframe></div>"
DEFAULT_PATH_TO_GRAPH = "/pycgi/graph.html?json_file={JSONPATH}&graph_config={GRAPH_CONFIG}"
PATH_TO_GRAPH_LAYOUTS = "/pycgi/dist/graphics/"

# Predefined GRAPH LAYOUT
TRENDLINE_GRAPHICS = PATH_TO_GRAPH_LAYOUTS + 'trendline.js'
LIGTHCURVE_GRAPHICS = PATH_TO_GRAPH_LAYOUTS + 'lightcurve.js'
IIIDLIGTHCURVE_GRAPHICS = PATH_TO_GRAPH_LAYOUTS + '3dlightcurve.js'

# Clear GRAPH CACHE
def clear_graph_cache(meteor_json_file_data,analysed_name,graph_type):
   make_plot(graph_type,meteor_json_file_data, analysed_name, True)
   
# Return an iframe with a graph or nothing if we don't have enough data
# for this graph type
def make_plot(graph_name,meteor_json_data,analysed_name,clear_cache):

   # Do we have a JSON ready this graph?
   path_to_json = get_graph_file(meteor_json_data,analysed_name,graph_name,clear_cache)
  
   if(path_to_json is None or cfe(path_to_json)==0):

      if(graph_name=="xy"):
         # Get the data
         if('frames' in meteor_json_data):
            if len(meteor_json_data['frames']) > 2:
               json_graph_content = create_xy_graph(meteor_json_data['frames'])
               
               if(json_graph_content is not None):
                  
                  # We save it at the right place
                  path_to_json = get_cache_path(analysed_name,"graphs")+graph_name+'.json'

                  # We save it
                  save_json_file(path_to_json,json_graph_content)

                  # Return the iframe code
                  return create_iframe_to_graph(analysed_name,graph_name,path_to_json,TRENDLINE_GRAPHICS)

      elif(graph_name=='curvelight'):
         # Get the data
         if('frames' in meteor_json_data):
            if len(meteor_json_data['frames']) > 2:
               json_graph_content = create_light_curve_graph(meteor_json_data['frames'])
               
               if(json_graph_content is not None):
                  
                  # We save it at the right place
                  path_to_json = get_cache_path(analysed_name,"graphs")+graph_name+'.json'

                  # We save it
                  save_json_file(path_to_json,json_graph_content)

                  # Return the iframe code
                  return create_iframe_to_graph(analysed_name,graph_name,path_to_json,LIGTHCURVE_GRAPHICS)

      elif(graph_name=='3Dlight'):
         if('frames' in meteor_json_data):
            if len(meteor_json_data['frames']) > 2:
               json_graph_content = create_3Dlight_curve_graph(meteor_json_data['frames'],analysed_name)
               
               if(json_graph_content is not None):
                  
                  # We save it at the right place
                  path_to_json = get_cache_path(analysed_name,"graphs")+graph_name+'.json'

                  # We save it
                  save_json_file(path_to_json,json_graph_content)
                  
                  # Return the iframe code
                  return create_iframe_to_graph(analysed_name,graph_name,path_to_json,IIIDLIGTHCURVE_GRAPHICS)


   else:
      if(graph_name=="xy"):
         return create_iframe_to_graph(analysed_name,graph_name,path_to_json,TRENDLINE_GRAPHICS)
      elif(graph_name=="curvelight"):
         return create_iframe_to_graph(analysed_name,graph_name,path_to_json,LIGTHCURVE_GRAPHICS)
      elif(graph_name=="3Dlight"):
         return create_iframe_to_graph(analysed_name,graph_name,path_to_json,IIIDLIGTHCURVE_GRAPHICS)
   return ""

# Build the iFrame 
# Create the corresponding JSON file for the Graph
# and create the iframe with file=this json
def create_iframe_to_graph(analysed_name,name,path_to_json,graph_config):
   # Create iframe src
   src =  DEFAULT_PATH_TO_GRAPH.replace('{JSONPATH}', path_to_json)
   src = src.replace('{GRAPH_CONFIG}',graph_config)

   return DEFAULT_IFRAME.replace('{CONTENT}',src)


# Get a graph.json or create it 
def get_graph_file(meteor_json_file,analysed_name,name,clear_cache):

    # CREATE or RETRIEVE TMP JSON FILE UNDER /GRAPH (see REDUCE_VARS)  
   json_graph = does_cache_exist(analysed_name,'graphs',name+'.json')
   path_to_json = None
 
   if  len(json_graph)==0  or clear_cache is True :
 

      # We need to create the JSON
      path_to_json = get_cache_path(analysed_name,"graphs")+name+'.json'
 
      # We delete the file  
      try:
         os.remove(path_to_json) 
      except:
         x=0 # Nothing here as if it fails, it means the file wasn't there anyway (?)
  
   else: 

      # We return them 
      path_to_json = glob.glob(get_cache_path(analysed_name,"graphs")+name+'.json') 

      if(path_to_json is not None and len(path_to_json)>0):
         path_to_json = path_to_json[0]
 

   return path_to_json
 



 
 

# Create BASIC x,y plot with regression (actually a "trending line")
def create_xy_graph(frames):
   # Do we have the json ready?
   xs = []
   ys = []
 
   for frame in frames:
      xs.append(frame['x']) 
      ys.append(frame['y']) 
 
   if(len(xs)>2):

      trend_x, trend_y = poly_fit_points(xs,ys)  
    
      tx1 = []
      ty1 = []

      for i in range(0,len(trend_x)):
         tx1.append(int(trend_x[i]))
         ty1.append(int(trend_y[i]))
  
      return   {'title':'XY Points and Trendline',
                'x1_vals':  xs,
                'y1_vals': ys,
                'x2_vals': tx1,
                'y2_vals': ty1,
                'y1_reverse':1,
                'title1': 'Meteor pos.',
                'title2': 'Trend. val.',
                's_ratio1':1} 
   return None

 
    

# Curve Light
def create_light_curve_graph(frames):
   lc_cnt = []
   lc_ff = []
   lc_count = []
   
   if(len(frames)>1):
      for frame in frames:
         if "intensity" in frame and "intensity_ff" in frame:
             if frame['intensity']!= '?' and frame['intensity']!= '9999':
               lc_count.append(frame['dt'][14:]) # Get Min & Sec from dt
               lc_cnt.append(frame['intensity']) 
               lc_ff.append(frame['intensity_ff']) 

      if(len(lc_count)>2 and len(lc_cnt)>2):
         return {
            'title':'Light Intensity',
            'title1': 'Intensity',
            'x1_vals':  lc_count,
            'y1_vals':  lc_cnt, 
            'linetype1': 'lines+markers',
            'lineshape1': 'spline'
         }
   return None


# 3D light Curve
def create_3Dlight_curve_graph(frames,analysed_name):
  return get_json_for_3Dlight_curve(frames,analysed_name)

 
 
# Compute the fit line of a set of data (MIKE VERSION)
def poly_fit_points(poly_x,poly_y, z = None):
   if z is None:
      if len(poly_x) >= 3:
         try:
            z = np.polyfit(poly_x,poly_y,1)
            f = np.poly1d(z)
         except:
            return 0
      else:
         return 0

      trendpoly = np.poly1d(z)
      new_ys = trendpoly(poly_x)

   return(poly_x, new_ys)
 





# Create 3D Light Curve Graph
def make3D_light_curve(meteor_json_file,hd_stack):
 
   xvals = []
   yvals = []
   zvals = []
 
   for x in range(0, HD_W):
      xvals.append(x)
   
   for y in range(0, HD_H):
      yvals.append(y)

   for z in range(0, 255):
      zvals.append(0)

   image = cv2.imread(hd_stack)

   for f in meteor_json_file['frames']:   
      try:
         #xvals.append(f['x'])
         #yvals.append(f['y'])
         zvals.append(statistics.mean(image[int(f['y']),int(f['x'])]))  # Average of the 3 VALUES
      except:
         partial = True
   
   if len(xvals)>0 and len(yvals)>0 and len(zvals)>0:
      return create_iframe_to_graph({
         'title':'3D Light Topography',
         'x1_vals': str(xvals),
         'y1_vals':str(yvals),
         'z1_vals':str(zvals) 
      })
   else:
      return ''

 