import pickle
import heapq
import math
from scipy.spatial import distance as dist
import time
import random
from shortestpath import find_path

def haversine(lat1, lon1, lat2, lon2):
  R = 6372800
  phi1, phi2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlambda = math.radians(lon2 - lon1)
  a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
  return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

with open("coordinates_final_dist", 'rb') as handle:
  dis_data = pickle.loads(handle.read())
with open("coordinates_final_safety", 'rb') as handle:
  safety_data = pickle.loads(handle.read())
with open("coordinates_final_traffic", 'rb') as handle:
  traffic_data = pickle.loads(handle.read())

print(list(traffic_data.keys())[0])

count = 0
temp = random.choices(list(dis_data.keys()), k = 1000)
for i in range(len(temp)):
  if haversine(temp[i][0], temp[i][1], 39.172619, -86.523384) > 500:
    print(temp[i])
    prev = time.time()
    print(find_path(dis_data, safety_data, traffic_data, (temp[i][0], temp[i][1]), (39.172619, -86.523384)))
    print(time.time()-prev)
    count += 1
    if count == 10:
      break
