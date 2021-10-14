import numpy as np
import cv2
import pdb
import os
from matplotlib import pyplot as plt

from sklearn import tree
import graphviz 


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

def f2(roi):
    """
    Calcular el nivel máximo de rojo de la conjuntiva
    
    Parameters
    ----------
    roi : numpy.ndarray (n x m x c) BGR | conjuntiva
    
    Returns
    -------
    float | nivel de rojo
    """
    return np.max(roi[:,:,2])

def f3(roi,mascara):
    """
    Canal a de imagen LAB
    
    Parameters
    ----------
    roi : numpy.ndarray (n x m x c) LAB | conjuntiva
    mascara : numpy.ndarray (n x m x c) | imagen binaria (0,1)
    
    Returns
    -------
    float | suma de A / área conjuntiva
    """
    return np.sum(roi[:,:,1])/len(np.where(mascara[:,:,0]==1)[0])

def f4(roi,mascara):
    return np.sum(np.abs(128-roi[:,:,0]))/len(np.where(mascara[:,:,0]==1)[0])

def clasificar(imagen,mascara,roi,venas):
    area=f1(mascara,venas)
    max_rojo = f2(roi)
    lab = f3(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB),mascara)
    hsv = f4(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),mascara)

    return [area,max_rojo,lab,hsv]

def evaluar(imagenes_n,etiquetas,json):
    """
    Evaluar el acierto en la clasificación. 
    
    Parameters
    ----------
    imagenes_n : [str] | nombre de la imagen
    etiquetas : [int]  | etiqueta asociada
    json : dict | nombres - etiquetas_reales

    Returns
    -------
    float | acierto 
    """
    sum = 0
    for im,et in zip(imagenes_n,etiquetas):
        comparacion = [x for x in json['imagenes'] if x['nombre'] == os.path.basename(im)]
        if et == comparacion[0]['clase']:
            sum +=1
    return sum/len(imagenes_n)

def arbol(X,Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    tree.plot_tree(clf)
    print(clf.predict(X))
    print(Y)

    dot_data = tree.export_graphviz(clf, filled=True, rounded=True,special_characters=True,class_names=['0','1','2'],out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("iris")