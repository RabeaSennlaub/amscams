import os

# Setup a live streaming on Youtube
VBR="1000k"                               
FPS="20"                              
QUAL="MEDIUM"  

 
# CAM IP
CAM_IP= "192.168.76.71"

# YOUTUBE Key
KEY="2e88-ec97-t36t-ec68"       

# Overlay Image
OVERLAY="/home/ams/amscams/dist/img/1280x720/AMS_UA.png"    
   
# Text
TEXT = "Cam Operator: Vishnu Reddy, UA"    


SOURCE="rtsp://"+CAM_IP+"/user=admin&password=&channel=1&stream=0.sdp"              # Source UDP (voir les annonces SAP)
YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2/"+KEY

cmd = 'ffmpeg \
      -ar 44100 -ac 2 -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -i "'+SOURCE+'" -i "'+OVERLAY+'" \
      -filter_complex "[1:v]scale=1280x720[scaled];\
      [scaled]drawtext=:text="\"'+TEXT+'\"":fontfile=\'/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf\':fontcolor=white@0.45:fontsize=14:x=20:y=20[texted]; \
      [texted]yadif[m];[m][2]overlay=25:25" \
      -vcodec libx264 -pix_fmt yuv420p -preset '+QUAL+' -r '+FPS+' -g $(('+FPS+'  * 2)) -b:v '+VBR+'  \
      -acodec libmp3lame -ar 44100 -threads 6 -qscale 3 -b:a 712000 -bufsize 512k \
      -f flv "'+YOUTUBE_URL+'"'

print(cmd)
os.system(cmd)