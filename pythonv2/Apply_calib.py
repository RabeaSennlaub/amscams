#!/usr/bin/python3 

import sys

from lib.MeteorManual_ApplyCalib import apply_calib

# JSON FILE 
json_file = sys.argv[1]          
new_json_content = apply_calib(json_file)

print("DONE")
print("NEW JSON:")
print(apply_calib)