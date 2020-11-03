# CV_final_project
Final project for IUB CSCI-657 Computer Vision


# Corresponding models and data
The pre-trained models and data can be found in this link;
https://drive.google.com/drive/folders/1QZCYzcJkwE7TnmllFkUc7ASfHA6URaWB
if you need access, please follow the request for access guideline by google, and we'll respond as soon as possible.

You can also train the models, using the corresponding model files;
side_walk.py, trafficlight.py, cross_walk_model.py, crosswalk_sign_model.py and stopsigns.py
Just remember to change the source directory of the imagedatagenerator to your data directory.

# How to run the project
The current default school is IU. If you want to, you can download different school's coordinates using CV_images_download.py, and you'll be able to run it on other schools also. If you do, you must create the three data matrices used by the main process by using

connected_graph_creation.py -> safety_py, traffic_data_creation.py

# Running the project through the frontend

The frontend code can be found under frontend folder in the link given below:
https://drive.google.com/drive/u/3/folders/1QZCYzcJkwE7TnmllFkUc7ASfHA6URaWB

```
Steps to run for setting up and running from the frontend:
  -> Download the frontend folder which contains Computer vision folder, httpServerCV.py and mainCode.py
  -> The front end given here is built using ReactJs. In order to run this, make sure you have a IDE which can run ReactJS framework.
      We have used WebStrom which is a JetBrains editor for ReactJS. 
  -> Install nodejs interpretor to run reactjs.
  -> Make sure your port 8081 is open and available.
  -> Open httpServerCV.py and do the following configurations in it:
      - Make sure the port is 8081 here.
      - Initialize the variable mainCodePath as path for mainCode.py
      - Provide your google key API in googleAPIKey variable. Also make sure geocode API is enabled for this key.
  -> Open mainCode.py and do the following configurations in it:
      - Provide the path for your co-ordinates_final_dist pickle file in the variable called pathForCoOrdinatesDictionary.
      - Provide the path for App.js in the varible called appJsFilePath.
      
  -> Run the httpServerCV.py first. This should start up and run in the given port.
  -> Run the reactjs code for frontend. 
```
