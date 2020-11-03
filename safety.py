"""safety.py

The function that saves the safety matrix

The images should be in directory
schoolname_image
with file name
Image + latitude + longitude + _degree of view .png

ex)
IU_image/Image39.170189,-86.52351597_270.png


"""


import numpy as np
from safety_score import get_score
import sys
import pickle
import os
import time

safety_data = {}
school_name = "IU"
image_names = {}

with open("coordinates_final_dist", 'rb') as handle:
  dis_data = pickle.loads(handle.read())

with open("sidewalk_result", 'rb') as handle:
  sidewalk = pickle.loads(handle.read())

with open("cross_sign_result", 'rb') as handle:
  cross_signs = pickle.loads(handle.read())

with open("crosswalks_result", 'rb') as handle:
  crosswalks = pickle.loads(handle.read())

with open("stop_sign_result", 'rb') as handle:
  stop_sign = pickle.loads(handle.read())

with open("traffic_light_result", 'rb') as handle:
  traffic_light = pickle.loads(handle.read())

print(list(sidewalk.keys())[0])
print(len(cross_signs.keys()))
print(list(crosswalks.keys())[0])
print(list(stop_sign.keys())[0])
print(list(traffic_light.keys())[0])
print(list(dis_data.keys())[0])

cnt = 0
flag = 0
for image_name in os.listdir(school_name + "_images/images"):
  temp = image_name.split('_')
  if temp[0] == 'Image':
    temp2 = temp[3].split('.')
    image_names[(temp[1], temp2[0])] = image_name
  else:
    coordinate = temp[0].split('Image')[1]
    angle = temp[1].split('.')[0]
    
    lat, lon = coordinate.split(',')
    lon_f, lon_dec = lon.split('.')
    for i in np.arange(6, 3, -1):
      temp = lon_dec[:i]
      if (float(lat), float(lon_f + "." + temp)) in dis_data.keys():
        flag = 1
        break
    if flag == 0:
      print((lat, lon_f + "." + temp))
    image_names[(lat + "," + lon_f + temp, angle)] = image_name
    
    

for coordinate in dis_data.keys():
  temp_score = get_score(school_name, coordinate, image_names,
                                      sidewalk, cross_signs,
                                      crosswalks, stop_sign,
                                      traffic_light)
  if temp_score == 0:
    cnt += 1
  safety_data[coordinate] = temp_score

with open('coordinates_final_safety', 'wb') as handle:
  pickle.dump(safety_data, handle)
print(cnt)
