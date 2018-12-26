import matplotlib.pyplot as plt
import matplotlib.image as impimg
import numpy as np
import cv2
import math
import time
from PIL import Image
import picamera
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    theta = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            theta = theta+math.atan2((y2-y1),(x2-x1))
    return img, theta

def valuesLeftRight(img):
    muestras = 0
    dist_izda = 0
    dist_dcha = 0

    for i in range(200, size[0] - 100, 20):

        izda = 0
        for x in (range(int(size[1] / 2), 0, -1)):
            if img[i][x] == 255:
                break
            else:
                izda = izda + 1
        dcha = 0
        for x in (range(int(size[1] / 2), size[1])):
            if img[i][x] == 255:
                break
            else:
                dcha = dcha + 1
        dist_izda = dist_izda + izda
        dist_dcha = dist_dcha + dcha
        muestras = muestras + 1

    dist_dcha = dist_dcha / muestras
    dist_izda = dist_izda / muestras

    return dist_izda, dist_dcha


#imagePath = 'test_images/curvadere.jpg'

#imagePath = 'test_images/curvaizquierda.jpg'
#imagePath = 'test_images/curvaderecha.jpg'
#imagePath = 'test_images/recta.jpg'
#imagePath = 'test_images/semibuena.jpg'
imagePath = 'test_images/semicurva.jpg'
camera = picamera.PiCamera()
time.sleep(1)

while(True):

    im= camera.capture('img.jpg')
    # Record previous state
    camera.close()
    #im = Image.open(imagePath)
    width, height = im.size
    region_of_interest_vertices = [
        (0, height),
        (0, height / 4),(width, height/4),
        (width, height),
    ]

    i = impimg.imread(imagePath)
    image = i[2:384, 0:5400]

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)


    #plt.figure()
    #plt.imshow(image)



    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    #plt.figure()
    #plt.imshow(cannyed_image)
    cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices],np.int32),)
    plt.figure()
    plt.imshow(cropped_image)
    size = cropped_image.shape
    #Cada 20 lineas, calcular distancia al 255 por la izquierda y por la derecha
    #izda = distizda + izda, lo mismo con la dcha (range(10, 0, -1))
    #ida/mediciones realizadas, lo mismo con la dcha
    #tomar decision

    dist_izda, dist_dcha = valuesLeftRight(cropped_image)
    print(cropped_image.shape)
    print(dist_izda, dist_dcha)

    if (abs(dist_izda-dist_dcha) >= 50):
        if(dist_izda>dist_dcha):
            print ('left')
        else:
            print('right')
    else:
        print('straight')

#    plt.show()
    time.sleep(1)
#recta: 472, 381
#curvadcha: 521, 511
#curvaizda:  593, 293
#semibuena: 435,426