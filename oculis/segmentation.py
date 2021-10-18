import numpy as np
import cv2
from matplotlib import pyplot as plt

LIMITE_ROJO = 145
MINIMO_MASCARAS = 4
N_MEDIAN = 4

def pinta_mascara(mascara,image):
    """
    Función auxiliar: Pinta la mascara sobre la imagen
    """
    contours, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 23, 0), 2, 8)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig("../borrar/segmentation_over/" + str(np.random.rand())+".png")

def bgr_to_tsl(image):  # bgr
    """
    Convertir imagen bgr a tsl. 
    Solo calcula los canales S y L
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR 
    Returns
    -------
    numpy.ndarray | imagen TSL 
    """
    image = image.astype(float)
    output = (image*0).astype(float)

    suma_ = (image[:,:,2]+image[:,:,1]+image[:,:,0])
    r = np.divide(image[:,:,2],suma_+0.00001)
    g = np.divide(image[:,:,1],suma_+0.00001)

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
    bloque_central1 = int(image.shape[0]/3)
    bloque_central2 = int(image.shape[1]/3)
    umbral = np.mean(image[bloque_central1:bloque_central1*2,bloque_central2:bloque_central2*2:,1])

    ix,iy = np.where(image[:,:,1]>=umbral)
    output[ix,iy]=1
    return output

def get_mtv(hsv_img):  # HSV
    """
    Thresholding en el canal V de una imagen HSV usando 
    la media del canal V
    
    Parameters
    ----------
    image : numpy.ndarray | imagen HSV
    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0 
    """
    output = hsv_img[:,:,0]*0
    umbral = np.mean(hsv_img[:,:,2])
    ix,iy = np.where(hsv_img[:,:,2]>=umbral)
    output[ix,iy]=1
    return output

def get_mtl(lab_img):  #LAB
    """
    Thresholding en el canal L de una imagen L*a*b* usando 
    la media del canal L
    
    Parameters
    ----------
    image : numpy.ndarray | imagen LAB
    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """
    output = lab_img[:,:,0]*0
    umbral = np.mean(lab_img[:,:,0])
    ix,iy = np.where(lab_img[:,:,0]>=umbral)
    output[ix,iy]=1
    return output

def get_mts(tsl_img):  #TSL
    """
    Thresholding en el canal S de una imagen TSL usando 
    la media del canal S
    Parameters
    ----------
    image : numpy.ndarray | imagen TSL
    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """

    m1 = tsl_img[:,:,0]*0
    umbral = np.mean(tsl_img[:,:,1])
    ix,iy = np.where(tsl_img[:,:,1]<=umbral)
    m1[ix,iy]=1
    return m1

def get_mts2(tsl_img,hsv_img):
    """
    Thresholding en el canal S de una imagen TSL usando 
    la media del canal S y considerando únicamente píxeles no rojos.
    
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR
    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """

    m1 = tsl_img[:,:,0]*0
    m2 = tsl_img[:,:,0]*0
    m3 = tsl_img[:,:,0]*0

    umbral = np.mean(tsl_img[:,:,1])
    ix,iy = np.where(tsl_img[:,:,1]<=umbral)
    m1[ix,iy]=1

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

def segmentar(image,post=True,pintar=False):
    """
    Identificación automática de la región del ojo a considerar.
    Parameters
    ----------
    image : numpy.ndarray | imagen BGR
    post: True | False | indica si realizar operaciones morfológicas para mejorar ROI
    pintar: True | False | indica si pintar máscara sobre imagen original y salvar resultado

    Returns
    -------
    numpy.ndarray | mascara (x,y) -> 1 o 0
    """

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    tsl_img = bgr_to_tsl(image)

    m=None
    m1=get_mtg(image)
    m2=get_mtg2(image)
    m3=get_mtv(hsv_img)
    m4=get_mtl(lab_img)
    m5=get_mts(tsl_img)
    m6=get_mts2(tsl_img,hsv_img)
    m7=get_mmo(image)
    
    bloque_central1 = int(image.shape[0]/3)
    bloque_central2 = int(image.shape[1]/3)
    nivel_rojo = np.mean(lab_img[bloque_central1:bloque_central1*2,bloque_central2:bloque_central2*2,1])
    #print(nivel_rojo)
    
    # Ponderacion de las distintas máscaras. m->2D entre 0 y 7
    if(nivel_rojo>LIMITE_ROJO):
        m = m3*3+m4*2+m6*2
    else:
        m = m1+m2+m3+m4+m5+m6+m7

    # Píxeles > MINIMO_MASCARAS 1; caso contrario 0.
    ii,jj=np.where(m>=MINIMO_MASCARAS)
    mascara = np.zeros((m.shape[0],m.shape[1]),dtype=np.uint8)
    mascara[ii,jj]=1

    # Depuración de la máscara a través de operaciones morfológicas
    if post:
        for i in range(N_MEDIAN):
            mascara = cv2.medianBlur(mascara, 17)
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23)) #np.ones((23,23), np.uint8)
        mascara = cv2.morphologyEx(mascara,cv2.MORPH_OPEN, kernel, iterations=6)

    if pintar: # pintar mascara sobre imagen para comparaciones
        pinta_mascara(mascara,image)

    # Generamos máscara de 3 canales triplicando aux (para poder multiplicar con imagen).
    aux = np.transpose(mascara)
    mascara = np.transpose(np.stack((aux,aux,aux)))
    return mascara