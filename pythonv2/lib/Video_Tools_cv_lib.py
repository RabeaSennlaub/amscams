import cv2
import numpy as np
import ImageFont
from lib.VIDEO_VARS import *   

# Get text position x,y
# based on Position = br, bl, tr, tl (ex: br = bottom right)
# and margins (distance of the overlay from the borders of the frame)
# return x,y position of the overlay over the background
def get_text_position_cv(background,text,position,font,margins=VIDEO_MARGINS):

    # Get font.getsize(txt)
    text_h, text_w = font.getsize(text)
    print("TEXT H " + str(text_h))
    print("TEXT W " + str(text_w))

    return 0,0

     
 
# Get overlay position x,y 
# based on Position = br, bl, tr, tl (ex: br = bottom right)
# and margins (distance of the overlay from the borders of the frame)
# return x,y position of the overlay over the background
def get_overlay_position_cv(background, overlay, position, margins=VIDEO_MARGINS):
    h, w = overlay.shape[0], overlay.shape[1] 
    if(position=='bl'):
        background_width,background_height  = background.shape[1], background.shape[0]
        return VIDEO_MARGINS,background_height-VIDEO_MARGINS-h
    elif(position=='tl'):
        background_width,background_height  = background.shape[1], background.shape[0]
        return background_width-VIDEO_MARGINS-w,VIDEO_MARGINS 
    elif(position=='br'):
        background_width,background_height  = background.shape[1], background.shape[0]
        return background_width-VIDEO_MARGINS-w,background_height-VIDEO_MARGINS-h        
    else:
        return VIDEO_MARGINS,VIDEO_MARGINS