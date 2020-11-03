"""CV_images_download.py

The program is used for downloading street view images
from google api.
"""

import urllib.request
import google_streetview.api
import google_streetview.helpers
import urllib
import random
import google_streetview.api
import google_streetview.helpers
from pygeodesy.ellipsoidalVincenty import LatLon
import pandas as pd
import math
from math import ceil, floor
import cv2
from PIL import Image, ImageOps
import os
import skimage.io
import skimage.transform
import skimage as sk
from skimage import util
from skimage import transform as tf

load_excel = pd.read_excel('final_coordinates.xlsx')
list_queue = []
degrees = [0,90,180,270]
string_name = "Image"
start = 0
end = 20000
secret_key = ''
path_to_images = "images/"
stored_image = path_to_images + 'gsv_0.jpg'


def googleMapsImagesDownload(string_name, secret_key, load_excel, degrees,  start, end, stored_image, path_to_images):
    """

    :param degrees: Contains the direction where street images need to be downloaded
    :param start: start of the csv file where co-ordinates are taken for downloading images
    :param end: end index of the csv file where co-ordinates are last taken
    :param stored_image: path for storing the images / folder name where images need to be downloaded and stored
    :return:
    """
    drop = 0
    for i in range(start, end):
        latitudelongitude = str(load_excel['latitude'][i]) + "," + str(load_excel['longitude'][i])
        j = 0
        try:
            for k in range(0, 4):
                apiargs = {
                    'location': latitudelongitude,
                    'size': '640x300',
                    'heading': str(degrees[j]),
                    'pitch': '0',
                    'key': secret_key
                }
                api_list = google_streetview.helpers.api_list(apiargs)
                results = google_streetview.api.results(api_list)
                results.download_links(path_to_images)
                file_name = ""
                file_name = path_to_images + string_name + "" + str(latitudelongitude) + "" + str(i) + "_" + str(
                    degrees[j]) + ".png"
                os.rename(stored_image, file_name)
                j = (j + 1) % 4
        except:
            drop += 1
            print("Exception occurs", drop)
            
            
def bingMapsImagesDownload(start, end, coOrdinatesCSVFileName, key):
    """

    :param start: start index for downloading images from csv
    :param end: end index for downloading images form csv
    :param coOrdinatesCSVFileName:  name of the csv file containing co-ordinates
    :return: Nothing
    """
    coordinates = [0, 1]
    headings = [0, 90, 180, 270]
    load_excel = pd.read_csv(coOrdinatesCSVFileName)
    j = 0
    links = ''
    for i in range(start, end):
        try:
            for k in range(0, 4):
                heading = headings[k]
                links = 'https://dev.virtualearth.net/REST/V1/Imagery/Map/Streetside/' + str(
                    load_excel['latitude'][i]) + "," + str(
                    load_excel['longitude'][i]) + '/0?mapSize=1200,1200&heading=' + str(
                    heading) + '&pitch=5&key='+key
                file_name = str(load_excel['latitude'][i]) + "," + str(load_excel['longitude'][i]) + "_" + str(
                    heading) + ".jpg"
                urllib.request.urlretrieve(links, file_name)
            print(i)
        except:
            print("Except", j)
            j += 1


googleMapsImagesDownload(string_name, secret_key, load_excel, degrees,  start, end, stored_image, path_to_images)
