"""shortestpath.py

Holds the find_path function that is used by safety.py
to find the best path(short & safe) according to the algorithm
from source to destination
"""

import pickle
import heapq
import math
from scipy.spatial import distance as dist
import time


def successors(start_coordinate, data):
  """
      find successors of the current coordinate
  """

  try:
      next_cordinate = data[start_coordinate]
      resultant_next_cordinate = []
      for key,value in next_cordinate.items():
          resultant_next_cordinate.append((value,key))


      return list(resultant_next_cordinate)
  except:
      return [(999, (999,999))]

def find_path(dis_data, safety_data, traffic_data, source, destination):
  """
      param
      dis_data : data that has distance between coordinates
      safety_data : data that has safety scores of each coordinate
      source : The starting coordinate of path
      destination : The destination coordinate of path

      return
      The overall walking distance and list of coordinates representing the path
  """
  heap = []
  visited_cordinates = []
  visited_cordinates.append(source)
  alternate_path = (999, 999, '')
  #max_dis = dist.euclidean(source, destination)
  heapq.heappush(heap, (0, source, str(source), 0))

  while (len(heap) > 0):
      (heuristics, start_coordinate, path, total) = heapq.heappop(heap)
      for (distance1, start_coordinate1) in successors(start_coordinate,dis_data):
          try:
              traffic = traffic_data[(start_coordinate[0], start_coordinate[1])][(start_coordinate1[0], start_coordinate1[1])]
              if traffic == "None":
                traffic_v = 0
              elif traffic == "Mild":
                traffic_v = 0.1
              else:
                traffic_v = 0.2
          except:
              traffic_v = 0
          temp = dist.euclidean(start_coordinate1, destination)
          if temp < alternate_path[0]:
              alternate_path = (temp, total + haversine(start_coordinate[0], start_coordinate[1], start_coordinate1[0], start_coordinate1[1]), path + " " + str(start_coordinate1))

          if distance1 == 999:
              break

          if start_coordinate1 == destination:
              result = (0, total + haversine(start_coordinate[0], start_coordinate[1], start_coordinate1[0], start_coordinate1[1]), path + " " + str(start_coordinate1))
              return result
          else:
              if start_coordinate1 not in visited_cordinates:

                  safety_v = safety_data[(start_coordinate[0], start_coordinate[1])]
                  dis_v = dis_data[(start_coordinate[0], start_coordinate[1])][(start_coordinate1[0], start_coordinate1[1])]
                  heapq.heappush(heap, (heuristics + 1 - safety_v +
                                 dis_v / 0.00621371 / 2 +
                                 traffic_v, start_coordinate1, path + " " + str(start_coordinate1), total + dis_v))
                  visited_cordinates.append(start_coordinate1)
  return alternate_path

def haversine(lat1, lon1, lat2, lon2):

  """
      takes two sets of coordinates and converts them into miles
  """
  R = 6372800
  phi1, phi2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlambda = math.radians(lon2 - lon1)
  a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
  ret_v =  2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  return ret_v * 0.000621371


