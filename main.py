import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import crop_image
import front_ocr
import back_ocr
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(prog='National-ID-Card-OCR',
                                 description='Extract Data from national id card')
parser.add_argument("-i", "--image", type=str, help="Image URL")
args = vars(parser.parse_args())

# load the image
image_url = args["image"]

#image = Image.open(requests.get(image_url, stream = True).raw)

# crop the image
image = crop_image.crop_img(image_url)

# check weather the image is front or back side

def image_classifier(image_file):
    global back_data, front_data
    #image = cv2.imread(image_file)
    original_image = image_file.copy()
    image = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resized = image.resize((256, 256))

    model = load_model('models/image_classifier.h5')

    output = model.predict(np.expand_dims(resized, 0))

    if output > .5:
        image = original_image
        data = front_ocr.front_data(image)
    else:
        image = original_image
        data = back_ocr.back_data(image)
    return data


data = image_classifier(image)

#front_result = front_ocr.front_data(front_image)
#back_result = back_ocr.back_data(back_image)
# cv2.imwrite('cropsy.jpg', image) #debugging
