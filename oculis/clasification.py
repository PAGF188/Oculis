import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
import os


def f1(mascara, venas):
    """
    Calcular el área ocupada por los vasos.
    
    Parameters
    ----------
    mascara : numpy.ndarray (n x m x c) | imagen binaria (0,1)
    venas : numpy.ndarray (n x m) | imagen binaria (0,255)
    
    Returns
    -------
    float | area ocupación vasos 
    """
    return(len(np.where(venas==255)[0])/len(np.where(mascara[:,:,0]==1)[0]))

def f2():
    return None

def clasificar(imagen,mascara,venas):
    area=f1(mascara,venas)
    m2 = f2()

    # logica de determinacion de etiqueta
    print(area)
    return 1

def evaluar(imagenes_n,etiquetas,json):
    sum = 0
    for im,et in zip(imagenes_n,etiquetas):
        comparacion = [x for x in json['imagenes'] if x['nombre'] == os.path.basename(im)]
        if et == comparacion[0]['clase']:
            sum +=1
    return sum/len(imagenes_n)
