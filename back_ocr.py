import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
from collections import namedtuple

#for windows users use the line below if the program environment didn't detect pytesseract as a package
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def back_data(back_image):
    image = Image.fromarray(back_image)
    resized = image.resize((1000, 600))
    image_array = np.array(resized)
    # convert image into grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # convert grayscale image into threshold
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3), np.uint8)
    cv2.imwrite("thresh-b.jpeg", thresh)

    ocr_location = namedtuple(
        "ocr_location", ["id", "bbox", "filter_keywords"])

    ocr_locations = [
        ocr_location("Degree", (300, 80, 520, 100), [""]),
        ocr_location("Gender", (660, 168, 150, 70), [""]),
        ocr_location("Religion", (510, 160, 140, 80), [""]),
        ocr_location("Status", (320, 160, 150, 80), [""]),
        ocr_location("Hus's Name", (300, 220, 500, 60), [""])]

    image = thresh
    for loc in ocr_locations:
        if (loc == ocr_locations[0]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('degree.png', roi) #debugging
            degree = pytesseract.image_to_string(
                rgb_image, lang="ara_combined+ara-amiri-3000", output_type=Output.STRING)
            degree = degree.replace('\n', ' ')
        elif (loc == ocr_locations[1]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('gender.png', roi) #debugging
            gender = pytesseract.image_to_string(
                rgb_image, lang="ara_combined+ara-amiri-3000", config='--psm 6', output_type=Output.STRING)
            gender = gender.replace('\n', '')
        elif (loc == ocr_locations[2]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('religion.png', roi) #debugging
            religion = pytesseract.image_to_string(
                rgb_image, lang="ara_combined+ara-amiri-3000", config='--psm 6', output_type=Output.STRING)
            religion = religion.replace('\n', '')
        elif (loc == ocr_locations[3]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            cv2.imwrite('status.png', roi) #debugging
            status = pytesseract.image_to_string(
                rgb_image, lang="ara_combined+ara-amiri-3000", output_type=Output.STRING)
            status = status.replace('\n', '')
        elif (loc == ocr_locations[4]):
            (x, y, w, h) = loc.bbox
            roi = image[y:y+h, x:x+w]
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            #cv2.imwrite('hus_name.png', roi) #debugging
            hus_name = pytesseract.image_to_string(
                rgb_image, lang="ara_combined+ara-amiri-3000", output_type=Output.STRING)
            hus_name = hus_name.replace('\n', '')

    output_file_json = {"degree": degree, "gender": gender,
                        "religion": religion, "status": status, "husName": hus_name}

    with open('backData.json', 'w', encoding='utf-8') as f:
        json.dump(output_file_json, f, ensure_ascii=False, indent=4)
