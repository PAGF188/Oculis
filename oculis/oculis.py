import cv2
from segmentation import * 
from enhancement import *
from clasification import *
import os
import sys
from matplotlib import pyplot as plt
import argparse
import pdb
import time
import json


# borrar
X = []
Y = [2,0,1,2,0,0,1,0,2,0]

# Globales:
imagenes = []        # nombre imágenes
imagenes_bgr = []    # imágenes np.array
segmentaciones = []  
resultado_vasos = []
etiquetas = []
output_directory = None
tiempo = 0
plt.rcParams["figure.figsize"] = [50,50]

# Parser --list
#     - Si es archivo -> almacenar para procesar.
#     - Si es directorio -> explorar y almacenar sus archivos para procesar (solo 1 nivel).
# Parser --output
#     - Si no existe lo creamos.
# Parser --evaluar
#     - Si se pasa archivo anotaciones json evaluamos etiquetas predichas con reales.

parser = argparse.ArgumentParser(description='Automatic grading of ocular hyperaemia')
parser.add_argument('-l','--list', nargs='+', help='<Required> Images to process', required=True)
parser.add_argument('-o','--output', help='<Required> Place to save results', required=True)
parser.add_argument('-e','--evaluar', help='<Optional> json to evaluate', required=False)
args = parser.parse_args()

for element in args.list:
    if os.path.isfile(element):
        imagenes.append(element)
    elif os.path.isdir(element):
        for f in os.listdir(element):
            if os.path.isfile(os.path.join(element,f)):
                imagenes.append(os.path.join(element,f))

output_directory = args.output
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)


# Procesamiento de cada imagen
print("%d %s" %(len(imagenes), "images are going to be processed...\n"))
print("%s |%s%s| %d/%d [%d%%] in %.2fs"  % ("Processing...","-" * 0," " * (len(imagenes)-0),0,len(imagenes),0,0),end='\r', flush=True)

i=1
for img in imagenes:
    print(img)
    imagen = cv2.imread(img)
    inicio = time.perf_counter() 

    #img = shine_removal(img)

    # Segmentación
    mascara=segmentar(imagen,True) 
    roi = imagen * mascara 

    # Vessel detection. Canny_blurred detector
    # vasos = cv2.morphologyEx(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.MORPH_GRADIENT, np.ones((2,2), np.uint8))
    vasos = histogram_eq(roi)
    vasos = cv2.GaussianBlur(vasos, (13,13), 0)
    vasos = cv2.Canny(cv2.cvtColor(vasos, cv2.COLOR_BGR2GRAY),5,30)  # 5 30
    #vasos = cv2.morphologyEx(vasos,cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)

    # Clasificación.
    features = clasificar(imagen,mascara,roi,vasos)
    X.append(features)


    fin = time.perf_counter()

    # Almacenamos resultados
    imagenes_bgr.append(imagen)
    segmentaciones.append(roi)      # resultado segmentación
    resultado_vasos.append(vasos)   # resultado identificación vasos
    #etiquetas.append(etiqueta)      # resultado clasificación
    tiempo += (fin-inicio)
    #print("%s |%s%s| %d/%d [%d%%] in %.2fs (eta: %.2fs)"  % ("Processing...",u"\u2588" * i," " * (len(imagenes)-i),i,len(imagenes),int(i/len(imagenes)*100),tiempo,(fin-inicio)*(len(imagenes)-i)),end='\r', flush=True)
    i+=1

print("\n")

arbol(X,Y)
exit()
# Evaluar si argumento json dado.
if args.evaluar is not None and os.path.isfile(args.evaluar):
    with open(args.evaluar) as f:
        data = json.load(f)
        acierto = evaluar(imagenes,etiquetas,data)
        print("Acierto:",acierto)

print("\n")
print("%s %s/" %("Saving results in",output_directory))

exit()
# Salvar resultados a figura
i=0
f, ax = plt.subplots(1,2)
for im,s,r in zip(imagenes_bgr,segmentaciones,resultado_vasos):
    vis = cv2.hconcat([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), r])
    cv2.imwrite(os.path.join(output_directory,str(i)+"edge5.png"), r) # vis
    #ax[0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # ax[0].imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    # plt.savefig(os.path.join(output_directory,str(i)+"edge1"))
    i+=1