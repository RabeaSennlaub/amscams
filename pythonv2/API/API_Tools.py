import json
import sys
import cgitb
import requests



def send_json(json_msg) {
   url = 'https://sleaziest-somali-2255.dataplicity.io/pycgi/webUI.py?cmd=API'        
   parameters = json_msg
   headers = {'content-type': 'application/json'} 
   response = requests.post(url, data = json_msg),headers=headers)
   print(response)
}


def send_error_message(msg): 
   cgitb.enable()
   send_json(json.dumps({'error':msg}))
   sys.exit(0)