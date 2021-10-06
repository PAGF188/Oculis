import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape


max_binary_value = 255

def colour_constancy(img):  #descartado
    for i in range(img.shape[2]):
        img[:,:,i] = (255/np.max(img[:,:,i]))*img[:,:,i]
    return img

def histogram_eq(img): #descartado

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def shine_removal(img):  
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mask = cv2.threshold(lab_img[:,:,0], 240, max_binary_value, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(src=mask, kernel=np.ones((3, 3)), iterations=5)
    ii,jj = np.where(mask!=0)
    img[ii,jj,:]=0
    return(img)