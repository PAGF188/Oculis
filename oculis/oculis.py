
import cv2
from segmentation import * 
from enhancement import *
from os import listdir
from matplotlib import pyplot as plt


files = [f for f in listdir("../dataset")]

imagenes = []
for f in files:
    imagenes.append(cv2.imread("../dataset/"+f))

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


