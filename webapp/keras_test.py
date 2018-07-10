import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def crop_image(image_path):
    im = Image.open(image_path).resize((400,400))
    width, height = im.size
    if height < 500 or width < 500:
        return(im)
    # if height>width:
    #     left = int((width - ((1/3)*height))/2)
    #     right = int(((width - ((1/3)*height))/2) + ((1/3)*height))
    #     top = int((1/3)*height)
    #     bottom = int((2/3)*height)
    # else:
    #     left = int((1/3)*width)
    #     right = int((2/3)*width)
    #     top = int((height - ((1/3)*width))/2)
    #     bottom = int(((height - ((1/3)*width))/2) + ((1/3)*width))
    # cropped = im.crop((left, top, right, bottom))
    # return (cropped)
    return(im)

def make_prediction(image):
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image, mode='tf')
    yhat = model.predict_classes(image)
    return(yhat)

def load_and_predict(image_path):
    tree = crop_image(image_path).resize((224,224))
    predict = make_prediction(tree)
    #print(predict[0])
    #print (types[predict[0]], plt.imshow(tree));
    return(types[predict[0]])

if __name__ == '__main__':
    #load_and_predict('./alder_1.jpg')
    print('Please import me to another script')
else:
    model = keras.models.load_model('model_vgg16_11.model')
    types = ['alder','cedar','dougfir','fir','maple','oak','pine']
