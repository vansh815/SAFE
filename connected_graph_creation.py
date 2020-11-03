from math import ceil, floor
import urllib.request
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import json
import math
import numpy as np
import heapq 
import pickle
from scipy.spatial import distance as dist

def haversine(lat1, lon1, lat2, lon2):
    R = 6372800
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def checkForPathGoogle(latitude1, longitude1, main_latitude, main_longitude,key):
    link = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+ str(latitude1)+","+str(longitude1)+"&destinations="+str(main_latitude)+","+str(main_longitude)+"&mode=walking&key="+key
    contents = urllib.request.urlopen(link).read()
    data = json.loads(contents)
    distance = data['rows'][0]['elements'][0]['distance']['text'].split()
    if distance[1] == 'ft':
        miles = np.float(np.float((distance[0]))/5280)
    else:
        miles=np.float(distance[0])
    return np.round(miles,4)


def checkForPathBing(latitude1,longitude1,main_latitude, main_longitude, key):
    link = "https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins="+str(latitude1)+","+str(longitude1)+"&destinations="+str(main_latitude)+","+str(main_longitude)+"&travelMode=walking&key="+key
    contents = urllib.request.urlopen(link).read()
    data = json.loads(contents)
    totlDistance=0
    totlDistance = (data['resourceSets'][0]['resources'][0]['results'][0]['travelDistance'])
    return totlDistance


def generateCoOrdinates(latitude, longitude, degrees, secret_key_google, main_latitude, main_longitude, key):
    """

    :param latitude: source latitude
    :param longitude: source longitude
    :param degrees: directions where co-ordinates should be taken for
    :param secret_key_google: secret key of google api
    :param main_latitude: main latitude of souce
    :param main_longitude: main longitude of source
    :param key: Google/Bing key
    :return: Dictionary of source and destination coordinates with distance
    """
    i = 0
    drops = 0
    m = 0
    hashset = set()
    priority_queue = [(0,main_latitude,main_longitude)]
    list_queue = [(0,0,0,0)]
    distance=0
    geo_distance = {}
    maxvalue = -99999.99
    final_input = []
    count = 0
    source_final = []
    dest_final = []
    dist_final = []
    while i < 100000 and len(list_queue) > 0:

        if i == 0:
            list_queue.pop(0)
        else:
            maxvalue, distance, latitude, longitude = list_queue.pop(0)
        latlong = str(latitude) + "," + str(longitude)
      
        
        j = 0
        for k in range(0, 4):
            p = LatLon(latitude, longitude)
            latitude1 = np.round(p.destination(10, degrees[j]).lat,6)
            longitude1 = np.round(p.destination(10, degrees[j]).lon,6)
            if maxvalue < 1000:
                count += 1
         
                total_distance = checkForPathGoogle(latitude1, longitude1, latitude, longitude,key)               
            
                if (latitude,longitude) in geo_distance:
                    geo_distance[(latitude,longitude)].update({(latitude1,longitude1):total_distance})
                else:
                    geo_distance[(latitude,longitude)] = {(latitude1,longitude1):total_distance}
                if (latitude1, longitude1) not in hashset:
                    maxvalue = haversine(main_latitude, main_longitude, latitude1,  longitude1)
                    hashset.add((latitude1, longitude1))
                    list_queue.append((maxvalue, total_distance, latitude1, longitude1))
         
                j = (j + 1) % 4
            else:
                drops += 1
                break
        i += 1
   
    return geo_distance


# Luddy Coordinates

main_latitude = 39.172619
main_longitude = -86.523384
latitude = main_latitude
longitude = main_longitude
degrees = [0,90,180,270]
secret_key_google = ''

geo_distance = generateCoOrdinates(latitude, longitude,  degrees, secret_key_google, main_latitude, main_longitude, secret_key_google)

with open('coordinates_final_dist', 'wb') as handle:
    pickle.dump(geo_distance, handle)

