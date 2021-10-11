
import cv2
from segmentation import * 
from enhancement import *
import os
from matplotlib import pyplot as plt
import argparse
import pdb


# Globales:

imagenes = []
output_directory = None

# Parser --list
#     - Si es archivo -> almacenar para procesar
#     - Si es directorio -> explorar y almacenar sus archivos para procesar (solo 1 nivel)
# Parser --output
#     - Si no existe lo creamos

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

exit()
plt.rcParams["figure.figsize"] = [50,50]
f, ax = plt.subplots(1,2)

for i in range(len(imagenes)):
    ax[0].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
    
    #imagenes[i] = shine_removal(imagenes[i])

    # Segmentacion
    mascara=segmentar(imagenes[i]) 
    imagenes[i] = imagenes[i]*mascara 

    # Vessel detection
    #imagenes[i] = cv2.GaussianBlur(imagenes[i], (11,11), 0)
    #imagenes[i] = cv2.Canny(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY),5,15)

    ax[1].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
    plt.savefig('../borrar/'+str(i)+"segmentation3")
# #pruebas eliminar brillos




# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


