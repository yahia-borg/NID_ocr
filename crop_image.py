import cv2


def crop_img(image):
    image_file = cv2.imread(image)
    image_file = cv2.resize(image_file, (1200, 900))
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)  # convert to grayscale

    # threshold to get just the paper
    thresh_gray = cv2.Canny(gray, 0, 255, apertureSize= 7)
    cv2.imwrite('Image_thresh.jpg', thresh_gray)  # debugging

    contours, hierarchy = cv2.findContours(
        thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)      # biggest bounding box so far
    mx_area = 300
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx

    # Crop and save
    roi = image_file[y:y+h, x:x+w]
    cv2.imwrite('newimage.jpg', roi)
    return roi
