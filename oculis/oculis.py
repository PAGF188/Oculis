import cv2
from segmentation import * 
from enhancement import *
from clasification import *
import os
from matplotlib import pyplot as plt
import argparse
import time
import json
from skimage.filters import threshold_local

# borrar
X = []
Y = [2,0,1,2,0,0,1,0,2,0]

# Globales:
imagenes = []         # nombre imágenes
imagenes_bgr = []     # imágenes np.array
segmentaciones = []   # mascara 0|1
resultado_vasos = []  # mascara 0|255
etiquetas = []        # 
output_directory = None
tiempo = 0
max_binary_value = 255
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
parser.add_argument('-m','--method', help='<canny | local_t> vessel localization methodology', 
                    type=str, default='canny', choices=['canny', 'local_t'])
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

metodo = args.method

# Procesamiento de cada imagen
print("%d %s" %(len(imagenes), "images are going to be processed...\n"))
print("%s |%s%s| %d/%d [%d%%] in %.2fs"  % ("Processing...","-" * 0," " * (len(imagenes)-0),0,len(imagenes),0,0),end='\r', flush=True)

i=1
for img in imagenes:
    print(img)
    imagen = cv2.imread(img)
    inicio = time.perf_counter() 

    # OBTENER MASCARA BRILLOS -----------------------------------
    mascara_brillos = shine_removal(imagen)

    # SEGMENTACIÓN ----------------------------------------------
    mascara = segmentar(imagen,True,False) 
    roi = imagen * mascara 

    # LOCALIZACIÓN DE VASOS -------------------------------------
    vasos = histogram_eq(roi)
    
    # # Aproximación 1: Canny_blurred detector
    if(metodo == 'canny'):
        vasos = cv2.GaussianBlur(vasos, (13,13), 0)
        vasos = cv2.Canny(cv2.cvtColor(vasos, cv2.COLOR_BGR2GRAY),5,30)  # 5 30
    # Aproximación 2: "Local threshold" con función propia.
    elif(metodo == 'local_t'):
        block_size = 21
        k=1
        func = lambda a: (a[len(a)//2]<=a.mean()+5 and a[len(a)//2]<=np.mean(a)-5).astype(int)*255
        vasos = threshold_local(cv2.cvtColor(vasos, cv2.COLOR_BGR2GRAY), block_size, 'generic', param=func)
        vasos = cv2.medianBlur(vasos.astype(np.uint8), 5)
        vasos = cv2.medianBlur(vasos.astype(np.uint8), 3)

    # CLASIFICACIÓN  -------------------------------------------
    vasos = vasos * mascara_brillos
    features = get_features(imagen,mascara,roi,vasos)
    etiqueta = predict(features)
    X.append(features)
    fin = time.perf_counter()

    # ALMACENAMOS RESULTADOS DE CADA ETAPA ---------------------
    imagenes_bgr.append(imagen)
    segmentaciones.append(roi)      # resultado segmentación
    resultado_vasos.append(vasos)   # resultado localización vasos
    etiquetas.append(etiqueta)      # resultado predicción
    tiempo += (fin-inicio)
    #print("%s |%s%s| %d/%d [%d%%] in %.2fs (eta: %.2fs)"  % ("Processing...",u"\u2588" * i," " * (len(imagenes)-i),i,len(imagenes),int(i/len(imagenes)*100),tiempo,(fin-inicio)*(len(imagenes)-i)),end='\r', flush=True)
    i+=1

print("\n")

arbol(X,Y)

# Evaluar si argumento json dado.
if args.evaluar is not None and os.path.isfile(args.evaluar):
    with open(args.evaluar) as f:
        data = json.load(f)
        acierto = evaluar(imagenes,etiquetas,data)
        print("Acierto:",acierto)

print("\n")
print("%s %s/" %("Saving results in",output_directory))

# Salvar resultados a figura
i=1
f, ax = plt.subplots(1,2)
for nombre,im,s,r in zip(imagenes,imagenes_bgr,segmentaciones,resultado_vasos):
    #vis = cv2.hconcat([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), r])
    cv2.imwrite(os.path.join(output_directory,os.path.basename(nombre)), r) # vis
    #cv2.imwrite(str(i)+".png", r) # vis
    #ax[0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # ax[0].imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    # plt.savefig(os.path.join(output_directory,str(i)+"edge1"))
    i+=1





"""
Notas:
    Alternativas consideradas a detectar vasos:

    1) 
    vasos = roi*0
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    ix,iy = np.where(lab_roi[:,:,1]>=140)
    vasos[ix,iy,:]=255


    2)
"""