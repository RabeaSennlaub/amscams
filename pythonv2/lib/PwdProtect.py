import datetime 
import cgitb
import json
import sys

KEY="AllonsEnfantsDeLaPatrie" 

def login_page():
    cgitb.enable()
    template = ""
    fpt = open("/home/ams/amscams/pythonv2/templates/login.html", "r")
    for line in fpt:
        template = template + line
    print(template)

# Decode String based on Key
def encode(key, string):
    encoded_chars = []
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr(ord(string[i]) + ord(key_c) % 256)
        encoded_chars.append(encoded_c)
    encoded_string = ''.join(encoded_chars)
    return encoded_string

# Encode String based on Key
def decode(key, string):
    encoded_chars = [] 
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr((ord(string[i]) - ord(key_c) + 256) % 256)
        encoded_chars.append(encoded_c)
    encoded_string = ''.join(encoded_chars)
    return encoded_string

def check_pwd_ajax(user, pwd):
    cgitb.enable()

    #check in conf/as6.json if pwd is right
    json_file = open('../../conf/as6.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    result = {}
    try:
        if(json_data['site']['ams_id']==user and json_data['site']['pwd']==pwd):
            result['passed'] = 1
        else:
           result['passed'] = 0
    except Exception:
        result['error'] = 'Password not found in your configuration. Please, contact Mike.'
    r = json.dumps(result)
    print(r)

    rz= encode(KEY,pwd)
    print(rz)


def setup_pwd(domain,user,pwd):

    #setup_pwd('www.moncul.com','toto','t0t0t')
    cgitb.enable()
    expiration = datetime.datetime.now() + datetime.timedelta(days=0.5)
    
    print("Content-type:text/html\r\n")
    print("Set-Cookie:User = "+user+";\r\n")
    print("Set-Cookie:Password = "+pwd+"\r\n")
    print("Set-Cookie:Expires = "+expiration.strftime("%a, %d-%b-%Y %H:%M:%S GMT")+";\r\n")
    print("Set-Cookie:Domain = lovable-falcon-4326.dataplicity.io;\r\n")
    print("Set-Cookie:Path = /;\n")
 
    
    #print "Set-Cookie:UserID = XYZ;\r\n"
    #print "Set-Cookie:Password = XYZ123;\r\n"
    #print "Set-Cookie:Expires = Tuesday, 31-Dec-2007 23:12:40 GMT";\r\n"
    #print "Set-Cookie:Domain = www.tutorialspoint.com;\r\n"
    #print "Set-Cookie:Path = /perl;\n"
    #print "Content-type:text/html\r\n\r\n"
