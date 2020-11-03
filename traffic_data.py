import urllib.request
import simplejson
import urllib.request
import json
import os
import sys

def trafficDataFor10Metres(bingApiKey, sourceLatitude, sourceLongitude, destinationLatitude, destinationLongitude):
    """
    
    :param bingApiKey: API key for BING routes API
    :param sourceLatitude: Latitude of the source
    :param sourceLongitude: Longitude of the source
    :param destinationLatitude: Latitude of the destination
    :param destinationLongitude: Longitude of the destination
    :return: Returns the traffic data which can have values from None, Mild, Moderate, Severe. It can return Unknown if the API cannot find traffic between the given points 
    """
    try:
        link = "http://dev.virtualearth.net/REST/V1/Routes?wp.0=" + str(sourceLatitude) + "," + str(sourceLongitude) + "&wp.1=" + str(destinationLatitude) + "," + str( destinationLongitude) + "&maxSolns=10&key="+str(bingApiKey)
        contents = urllib.request.urlopen(link).read()
        data = json.loads(contents)
    except:
        return "Unknown"
    
    return data['resourceSets'][0]['resources'][0]['trafficCongestion']
