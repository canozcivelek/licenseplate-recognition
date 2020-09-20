# Import necessary libraries.
import os
import cv2
import numpy as np
import pytesseract

# Define folder names.
folder_in = 'images/'
folder_out = 'final/'

# Point script to tesseract exe.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define counter to name the output images.
cnt = 0

# Iterate through all images inside input directory.
for filename in os.listdir(folder_in):
    cnt += 1

    # Read image and resize all images to have a size of 520x120 (standard plate size for Turkey)
    image = cv2.imread(os.path.join(folder_in, filename))
    resized = cv2.resize(image, (520, 120), cv2.INTER_LINEAR)
    # Convert image to grayscale, apply threshold, blur to get rid of noise.
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    # Detect edges to be able to find cobtours in image.
    canny = cv2.Canny(blur, 30, 190)

    # Find and draw contours to isolate the plate from its surroundings.
    _, cnts, _ = cv2.findContours(canny, cv2.RETR_LIST  , cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by area to filter out the best possible matches.
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    # image_copy = resized.copy()
    # _ = cv2.drawContours(image_copy, cnts, -1, (0,0,255),2)

    # First write the original resized images into output directory.
    cv2.imwrite(f"final/plate{cnt}.jpg", resized)

    # Loop through contours that have 4 sides and an area of more than 6000.
    plate = None
    for c in cnts:
        # print('Area:', cv2.contourArea(c))
        perimeter = cv2.arcLength(c, True)
        # Approximate any 4-sided poly to a reectangle.
        edge_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        # If a 4-sided contour that has an area of more than 6000 has been found
        # overwrite the original image with a properly cropped image.
        if len(edge_count) == 4 and cv2.contourArea(c) > 6000:
            x,y,w,h = cv2.boundingRect(c)
            plate = resized[y:y+h, x:x+w]
            # Overwrite.
            cv2.imwrite(f"final/plate{cnt}.jpg", plate)

# Iterate through final versions of images.
for filename in os.listdir(folder_out):
    image = cv2.imread(os.path.join(folder_out, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Provide Tesseract with an input image.
    text = pytesseract.image_to_string(image, config='--psm 13')
    # Get rid of the last two digits from origina tesseract output.
    text_revised = text[:-2]
    # Check if the first and last digit of the plate has been detected as a digit.
    if text_revised[0].isdigit() and text_revised[-1].isdigit():
        if len(text_revised) >= 8:
            print(text_revised)

    # If not, get rid of the non-digit digits.
    if not text_revised[0].isdigit() or not text_revised[-1].isdigit():
        if len(text_revised) >= 8:
            text2 = text_revised[1:-1]
            print(text2)

    # Display images.
    cv2.imshow('final', image)
    cv2.waitKey()

cv2.destroyAllWindows()





























#
