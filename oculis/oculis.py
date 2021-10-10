
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
    imagenes[i] = cv2.GaussianBlur(imagenes[i], (11,11), 0)
    imagenes[i] = cv2.Canny(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2GRAY),1,10)

    ax[1].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
    plt.savefig('../borrar/'+str(i)+"edge")
# #pruebas eliminar brillos



#resultado1 = get_mtg(bgr_img)
#resultado2 = get_mtg2(bgr_img)
#resultado3 = get_mtv(bgr_img)
#resultado4 = get_mtl(bgr_img)

#cv2.imwrite('./resultado1.jpg', resultado1)
#cv2.imwrite('./resultado2.jpg', resultado2)
#cv2.imwrite('./resultado3.jpg', resultado3)
#cv2.imwrite('./resultado4.jpg', resultado4)




# cv2.imshow('Truncated Threshold', resultado1)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


