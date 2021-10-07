# M T G , M T G ′ , M T S , M T S ′ , M T V , M T L |  M M G , M M O , M W and M SM .

import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt

LIMITE_ROJO = 140
LIMITE_SATURACION = 130

def bgr_to_tsl(image):  # bgr
    
    image = image.astype(float)
    output = (image*0).astype(float)

    suma_ = (image[:,:,2]+image[:,:,1]+image[:,:,0])
    r = np.divide(image[:,:,2],suma_+0.0001)
    g = np.divide(image[:,:,1],suma_+0.0001)

    r_prima = r - 1/3
    g_prima = g - 1/3
    
    # El canal T no nos interesa.

    # Canal S
    output[:,:,1] = (np.sqrt((9/5)*(np.power(r_prima,2)+np.power(g_prima,2))))*255
    
    # Canal L
    output[:,:,2] = 0.299*image[:,:,2]+0.587*image[:,:,1]+0.114*image[:,:,0]

    return(output.astype(np.uint8))

def get_mtg(image):
    """
    Thresholding en el canal G de una imagen BGR usando 
    la media de la intensidad de toda la imagen
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR 

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0 
    """
    output = image[:,:,0]*0
    umbral = np.mean(image)
    ix,iy = np.where(image[:,:,1]>=umbral)
    output[ix,iy]=1
    return output

def get_mtg2(image):
    """
    Thresholding en el canal G de una imagen BGR usando 
    la media del tronco central del canal G
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0 
    """
    output = image[:,:,0]*0
    bloque_central = int(image.shape[0]/3)
    umbral = np.mean(image[bloque_central:bloque_central*2,:,1])

    ix,iy = np.where(image[:,:,1]>=umbral)
    output[ix,iy]=1
    return output

def get_mtv(image): 
    """
    Thresholding en el canal V de una imagen HSV usando 
    la media del canal V
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0 
    """
    output = image[:,:,0]*0
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    umbral = np.mean(hsv_img[:,:,2])
    ix,iy = np.where(hsv_img[:,:,2]>=umbral)
    output[ix,iy]=1
    return output

def get_mtl(image):
    """
    Thresholding en el canal L de una imagen L*a*b* usando 
    la media del canal L
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """
    output = image[:,:,0]*0
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    umbral = np.mean(lab_img[:,:,0])
    ix,iy = np.where(lab_img[:,:,0]>=umbral)
    output[ix,iy]=1
    return output

def get_mts(image):
    """
    Thresholding en el canal S de una imagen TSL usando 
    la media del canal S

    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """

    m1 = image[:,:,0]*0

    tsl_img = bgr_to_tsl(image)
    umbral = np.mean(tsl_img[:,:,1])
    ix,iy = np.where(tsl_img[:,:,1]<=umbral)
    m1[ix,iy]=1
    return m1

def get_mts2(image):
    """
    Thresholding en el canal S de una imagen TSL usando 
    la media del canal S considerando unicamente píxeles no rojos.
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """

    m1 = image[:,:,0]*0
    m2 = image[:,:,0]*0
    m3 = image[:,:,0]*0

    tsl_img = bgr_to_tsl(image)
    umbral = np.mean(tsl_img[:,:,1])
    ix,iy = np.where(tsl_img[:,:,1]<=umbral)
    m1[ix,iy]=1

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ix,iy = np.where(np.logical_or(np.logical_and(hsv_img[:,:,0]>=213,hsv_img[:,:,0]<=255),np.logical_and(hsv_img[:,:,0]>=0,hsv_img[:,:,0]<=21)))
    m2[ix,iy] = 1

    ix,iy = np.where(np.logical_and(np.logical_and(hsv_img[:,:,2]>=96,hsv_img[:,:,2]<=255),np.logical_and(hsv_img[:,:,1]>=96,hsv_img[:,:,1]<=255)))
    m3[ix,iy] = 1
    
    return(np.logical_or(m1,m2*m3))

def get_mmo(image,iter=10):
    """
    Thresholding en el canal G de una imagen BGR usando 
    la media de la intensidad de toda la imagen. Seguido
    de "morphological openings".
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """
    output = image[:,:,0]*0
    umbral = np.mean(image)
    ix,iy = np.where(image[:,:,1]>=umbral)
    output[ix,iy]=1
    kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    output = cv2.morphologyEx(output, op=cv2.MORPH_OPEN, kernel=kernel_, iterations=iter)
    return output

def segmentar(image):
    
    m=None
    m1=get_mtg(image)
    m2=get_mtg2(image)
    m3=get_mtv(image)
    m4=get_mtl(image)
    m5=get_mts(image)
    m6=get_mts2(image)
    m7=get_mmo(image)


    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bloque_central = int(image.shape[0]/3)
    nivel_rojo = np.mean(lab_img[bloque_central:bloque_central*2,:,1])
    print(nivel_rojo)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    nivel_saturacion = np.max(hsv_img[:,:,1])
    print(nivel_saturacion)
    # rojo intenso -> m4,m3,m6 mejores
    if(nivel_rojo>LIMITE_ROJO):
        m = m3*3+m4*2+m6*2
    # imagenes muy claras
    elif(nivel_saturacion>LIMITE_SATURACION):
        m = m1*3+m2*2+m5+m7
    else:
        m = m1+m2+m3+m4+m5+m6+m7
    print("\n")
    return m
        
        