import pdb
import numpy as np
import cv2

max_binary_value = 255

def power_law(im,gamma):      #descartado
    im = im.astype(float)
    min_im, max_im = np.min(im), np.max(im)          
    im = (im - min_im) / (max_im - min_im)
    return (255*np.power(im,gamma)).astype(np.uint8)

def adjust_gamma(image, gamma=1.0):  #descartado
	table = np.array([((i / 255.0) ** gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def prueba(img):  #descartado
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if np.mean(lab_img[:,:,0])<100:
        img = adjust_gamma(img,0.5)
    return(img)
    

def colour_constancy(img):  #descartado
    img = (255/np.max(img))*img
    return img

def histogram_eq(img):

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def shine_removal(img):  
    aux = np.ones((img.shape[0],img.shape[1]))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mask = cv2.threshold(lab_img[:,:,0], 240, max_binary_value, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(src=mask, kernel=np.ones((3, 3)), iterations=5)
    ii,jj = np.where(mask!=0)
    aux[ii,jj]=0
    return(aux)