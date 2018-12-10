import matplotlib.image as impimg
import cv2
import numpy as np
import math
from PIL import Image

#cortar imagen
def crop_image(image):
    row = 4
    col = 6
    img = image.copy()
    img = img[row:-row, col:-col, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

#estandarizar
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    cropped = crop_image(standard_im)
    return cropped

#divir (solo nos haria falta el de arriba y el de abajo)
def slice_image(image):
    img = image.copy()
    upper = img[0:7, :, :]
    #middle = img[8:15, :, :]
    lower = img[16:24, :, :]
    return upper, lower

#ver cual esta encendido
def get_avg_v(rgb_image):
    feature = [0, 0, 0]
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # calculate image area
    area = hsv.shape[0] * hsv.shape[1]

    # Add up all the pixel values in the V channel
    sum_v = np.sum(hsv[:, :, 2])
    avg_v = sum_v / area

    return math.floor(avg_v)

imagePath = "pathImage"
i = impimg.imread(imagePath)
cropped = crop_image(imagePath)
standarized = standardize_input(cropped)
upper, lower = slice_image(standarized)
upper_hv = get_avg_v(upper)
lower_hv = get_avg_v(lower)
if(upper_hv>lower_hv):
    print('red')
else:
    print('green')