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

def f5(roi,venas):
    """
    Calcular el nivel de rojo de los vasos.
    
    Parameters
    ----------
    roi : numpy.ndarray (n x m x c) | LAB | conjuntiva
    venas : numpy.ndarray (n x m x c) | imagen binaria (0,255)
    
    Returns
    -------
    float | suma roi * venas (solo píxeles de vasos) / área vasos
    """
    min, max = np.min(venas), np.max(venas)          
    aux = (venas - min) / (max - min)
    aux = aux.astype(np.uint8)
    return np.sum(roi[:,:,1]*venas)/len(np.where(aux==1)[0])

def get_features(imagen,mascara,roi,venas):
    area = f1(mascara,venas)
    rojo_vasos = f5(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB),venas)
    #print(("%.4f  %.4f")%(area,rojo_vasos))
    return [area,rojo_vasos]

def predict(features, metodo):
    if metodo == 'canny':
        if features[1]<=36106:      # si no es muy rojo -> 0 o 1
            if features[1]<=34730: 
                return 0
            else:
                if features[0]<=0.066:  # poca area -> 0
                    return 0
                else:                   # mucha area -> 1
                    return 1
        else:  # si muy rojo -> 2
            return 2
    elif metodo == 'local_t':
        if features[0] <= 0.23:
            return 0
        else:
            if features[1] <= 35181:
                return 0
            else:
                if features[1] <= 36492:
                    return 1
                else:
                    return 2
            

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
    #print(clf.predict(X))
    #print(Y)

    dot_data = tree.export_graphviz(clf, filled=True, rounded=True,special_characters=True,class_names=['0','1','2'],out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("iris")