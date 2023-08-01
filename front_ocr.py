import pytesseract
from pytesseract import Output
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import json
from collections import namedtuple
import re

#for windows users use the line below if the program didn't detect pytesseract as a package
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_arabic_text(text):
    text = re.sub(r"[^\0u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text

def front_data(front_image):
    image = Image.fromarray(front_image)
    resized = image.resize((1000, 600))
    image_array = np.array(resized)
    kernel1 = np.array([[-1,-1,-1],[-1,11,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(image_array,-1, kernel1)
    cv2.imwrite("sharp.jpeg", sharpened)
    # convert image into grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    # convert grayscale image into threshold (pure black & white)
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("thresh.jpeg", thresh)
    # kernel = np.ones((5,5), np.uint8)

    ocr_location = namedtuple(
        "ocr_location", ["id", "bbox", "filter_keywords"])

    ocr_locations = [
        ocr_location("First name", (260, 145, 720, 140), [""]),
        ocr_location("Address", (265, 270, 700, 140), [""]),
        ocr_location("NID", (380, 425, 600, 100), [""]),
        ocr_location("Factory", (60, 520, 300, 70), [""])]
    image = thresh
    for loc in ocr_locations:
        if (loc == ocr_locations[0]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('name.png', rgb_image) #debugging
            full_name = pytesseract.image_to_string(
                rgb_image, lang='ara_combined', config="--psm 12",output_type=Output.STRING)
            full_name = full_name.replace('\n', ' ')
        elif (loc == ocr_locations[1]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('address.png', roi) #debugging
            address = pytesseract.image_to_string(
                rgb_image, lang='ara_combined+ara_number', config='--psm 11', output_type=Output.STRING)
            address = address.replace('\n', ' ')
            address = clean_arabic_text(address)
        elif (loc == ocr_locations[2]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('national-ID.png', rgb_image)  #debugging
            NID = pytesseract.image_to_string(
                rgb_image, lang='ara_number+ara_combined',output_type=Output.STRING)
            NID = NID.replace('\n', '')
        elif (loc == ocr_locations[3]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('factory.png', roi)  #debugging
            factory = pytesseract.image_to_string(
                rgb_image, lang='eng+ara_number', output_type=Output.STRING)
            factory = factory.replace('\n', '')

    if NID[0] == '2':
        year = '19' + NID[1:3]
    else:
        year = '20' + NID[1:3]
    month = NID[3:5]
    day = NID[5:7]
    DOB = day + '/' + month + '/' + year

    output_json_file = {"fullName": full_name, "address": address,
                        "NID": NID, "DOB": DOB, "factory": factory}
    # return output_json_file

    with open('frontData.json', 'w', encoding='utf-8')as f:
        json.dump(output_json_file, f, ensure_ascii=False, indent=4)
