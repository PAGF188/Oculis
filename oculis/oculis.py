import cv2
from segmentation import * 
from enhancement import *
import os
import sys
from matplotlib import pyplot as plt
import argparse
import pdb
import time
import clasification


# Globales:
imagenes = []
imagenes_bgr = []
resultados = []
segmentaciones = []
output_directory = None
tiempo = 0
plt.rcParams["figure.figsize"] = [50,50]

# Parser --list
#     - Si es archivo -> almacenar para procesar.
#     - Si es directorio -> explorar y almacenar sus archivos para procesar (solo 1 nivel).
# Parser --output
#     - Si no existe lo creamos.

parser = argparse.ArgumentParser(description='Automatic grading of ocular hyperaemia')
parser.add_argument('-l','--list', nargs='+', help='<Required> Images to process', required=True)
parser.add_argument('-o','--output', help='<Required> Place to save results', required=True)
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
    imagen = cv2.imread(img)
    output = imagen*1
    inicio = time.perf_counter() 

    #img = shine_removal(img)

    # Segmentacion
    mascara=segmentar(imagen,True) 
    output = imagen * mascara 
    segmentaciones.append(imagen*mascara)

    # Vessel detection. Canny_blurred detector
    # output = cv2.morphologyEx(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), cv2.MORPH_GRADIENT, np.ones((2,2), np.uint8))
    output = histogram_eq(output)
    output = cv2.GaussianBlur(output, (11,11), 0)
    output = cv2.Canny(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),5,30)
    output = cv2.morphologyEx(output,cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)

    #Clasificacion. Features
    clasifica(output,imagen)

    fin = time.perf_counter()

    imagenes_bgr.append(imagen)
    resultados.append(output)
    tiempo += (fin-inicio)
    print("%s |%s%s| %d/%d [%d%%] in %.2fs (eta: %.2fs)"  % ("Processing...",u"\u2588" * i," " * (len(imagenes)-i),i,len(imagenes),int(i/len(imagenes)*100),tiempo,(fin-inicio)*(len(imagenes)-i)),end='\r', flush=True)
    i+=1

print("\n")
print("%s %s/" %("Saving results in",output_directory))

i=0
f, ax = plt.subplots(1,2)
for im,s,r in zip(imagenes_bgr,segmentaciones,resultados):
    vis = cv2.hconcat([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), r])
    cv2.imwrite(os.path.join(output_directory,str(i)+"edge4.png"), vis)
    #ax[0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # ax[0].imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    # plt.savefig(os.path.join(output_directory,str(i)+"edge1"))
    i+=1