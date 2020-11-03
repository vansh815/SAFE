"""safety_score.py

This function calculates the safety score of the coordinate
according to the models.

Image size should be 640x300

The current default models are;

side_walk : side_walk_model.h5
cross_signs : cross_sign_model.h5
crosswalks : crosswalks_model.h5
stop signs : stop_sign_model.h5
traffic lights = traffic_light_model.h5

"""


import tensorflow as tf
import numpy as np
import cv2

# The score functions return the safety score estimate
# calculated by each model
def sidewalk_score(sidewalk, images):
  prediction_0 = np.asarray(sidewalk[images[0]])
  prediction_90 = np.asarray(sidewalk[images[1]])
  prediction_180 = np.asarray(sidewalk[images[2]])
  prediction_270 = np.asarray(sidewalk[images[3]])
  
  conf_0 = max([prediction_0[0], prediction_90[0], prediction_180[0], prediction_270[0]])
  conf_1 = max([prediction_0[1], prediction_90[1], prediction_180[1], prediction_270[1]])
  conf_2 = max([prediction_0[2], prediction_90[2], prediction_180[2], prediction_270[2]])

  if conf_0 > conf_1 and conf_0 > conf_2:
    score = (1 - conf_0)
  elif conf_1 >= conf_0 and conf_1 > conf_2:
    if conf_2 >= conf_0:
      score = 2 - conf_1
    else:
      score = conf_1
  else:
    score = 1 + conf_2

  return score

def crosswalk_score(crosswalks, images):
  prediction_0 = np.asarray(crosswalks["IU_images/" + images[0]])
  prediction_90 = np.asarray(crosswalks["IU_images/" + images[1]])
  prediction_180 = np.asarray(crosswalks["IU_images/" + images[2]])
  prediction_270 = np.asarray(crosswalks["IU_images/" + images[3]])
  
  conf_0 = max([prediction_0[0], prediction_90[0], prediction_180[0], prediction_270[0]])
  conf_1 = max([prediction_0[1], prediction_90[1], prediction_180[1], prediction_270[1]])
 
  return conf_1

def cross_signs_score(cross_signs, images):
  prediction_0 = np.asarray(cross_signs["IU_images/" + images[0]])
  prediction_90 = np.asarray(cross_signs["IU_images/" + images[1]])
  prediction_180 = np.asarray(cross_signs["IU_images/" + images[2]])
  prediction_270 = np.asarray(cross_signs["IU_images/" + images[3]])
  
  conf_0 = max([prediction_0[0], prediction_90[0], prediction_180[0], prediction_270[0]])
  conf_1 = max([prediction_0[1], prediction_90[1], prediction_180[1], prediction_270[1]])
  
  return conf_1

def stop_score(stop_sign , traffic_light, images):
  # calculate score
  prediction_t_0 = np.asarray(traffic_light["IU_images/" + images[0]])
  prediction_t_90 = np.asarray(traffic_light["IU_images/" + images[1]])
  prediction_t_180 = np.asarray(traffic_light["IU_images/" + images[2]])
  prediction_t_270 = np.asarray(traffic_light["IU_images/" + images[3]])
  
  prediction_0 = np.asarray(stop_sign["IU_images/" + images[0]])
  prediction_90 = np.asarray(stop_sign["IU_images/" + images[1]])
  prediction_180 = np.asarray(stop_sign["IU_images/" + images[2]])
  prediction_270 = np.asarray(stop_sign["IU_images/" + images[3]])
  
  conf_t_1 = max([prediction_t_0[1], prediction_t_90[1], prediction_t_180[1], prediction_t_270[1]])
  conf_s_1 = max([prediction_0[1], prediction_90[1], prediction_180[1], prediction_270[1]])  
  
  return conf_t_1 , conf_s_1

# The model names are just included randomly
# Please change your model names
# Please change it in the file description also
def get_score(school_name, coordinate,
              image_names,
              sidewalk,
              cross_signs,
              crosswalks,
              stop_sign,
              traffic_light):

  images = load_image(school_name, coordinate, image_names)
  if images is None:
    return 0

  trafficscore, stopscore = stop_score(stop_sign, traffic_light, images)

  return ((sidewalk_score(sidewalk, images)/2)**2 +
          crosswalk_score(crosswalks, images)**2 +
          cross_signs_score(cross_signs, images)**2+
          trafficscore**2 + stopscore**2) / 5

def load_image(school_name, coordinate, image_names):
  try:

    img_0 = image_names[(str(coordinate[0])+","+str(coordinate[1]), str(0))]
        
    img_90 = image_names[(str(coordinate[0])+","+str(coordinate[1]), str(90))]

    img_180 = image_names[(str(coordinate[0])+","+str(coordinate[1]), str(180))]
        
    img_270 = image_names[(str(coordinate[0])+","+str(coordinate[1]), str(270))]

    return img_0, img_90, img_180, img_270
  except:
    return None
  
