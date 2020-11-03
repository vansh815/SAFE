from PIL import Image, ImageOps
import os

def scale_image(img, width, height):
    """

    :param img: Image to be scaled
    :param width: required width of the image
    :param height: Required height of the image
    :return: scaled image according to given height and width
    """

    img.thumbnail((width,height))
    left = 0
    right = 0
    top = 0
    bottom = 0
    
   
    diff = width-img.size[0]
    if diff == 0 or diff%2==0:
        left = int(diff/2)
        right = int(diff/2)
    else:
        left = int(diff/2)
        right = int(diff/2)+1

        
    diff = height-img.size[1]
    if diff == 0 or diff%2==0:
        top = int(diff/2)
        bottom = int(diff/2)
    else:
        top = int(diff/2)
        bottom = int(diff/2)+1

        
    return ImageOps.expand(img, border= (left, top, right, bottom))    

images_dir = 'data/zebracrossing_new'
save_dir = 'data/zebracrossing_newpadded/'

images = []
for file in os.listdir(images_dir):
    images.append(images_dir+ "/" +file)


for i in range(len(images)):
    img = Image.open(images[i])
    file_name = 'zebracross_class1_'+str(i)+'.png'
    scale_image(img,640,300).save(save_dir +file_name)
