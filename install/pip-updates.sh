sudo apt-get install python3-mpltoolkits.basemap
sudo apt install imagemagick
sudo pip3 install suntime
sudo pip3 install flask
sudo pip3 install psutil
sudo pip3 install uwsgi
#cartopy
sudo pip3 install --upgrade cython numpy pyshp six
sudo pip3 install shapely --no-binary shapely
sudo apt-get install libgeos-dev
sudo apt-get install libproj-dev
sudo pip3 install geopy
sudo pip3 install cartopy
sudo pip3 install pymap3d
sudo apt-get install redis-server

# REDIS SETUP
#supervised systemd
#bind 127.0.0.1 ::1
#vi /etc/redis/redis.conf 
#sudo systemctl restart redis.service


# edit conf add service
pip3 install redis

