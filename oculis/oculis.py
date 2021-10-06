
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

bgr_to_tsl(imagenes[0])

for i in range(len(imagenes)):
    ax[0].imshow(cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB))
    
    # Comienza procesamiento de la imagen
    
    imagenes[i] = cv2.GaussianBlur(imagenes[i], (11,11), 0)
    imagenes[i] = shine_removal(imagenes[i])

    m1=get_mtg(imagenes[i])
    m2=get_mtg2(imagenes[i])
    m3=get_mtv(imagenes[i])
    m4=get_mtl(imagenes[i])
    m5=get_mts(imagenes[i])
    m6=get_mts2(imagenes[i])
    m7=get_mmo(imagenes[i])
    m8=get_mg(imagenes[i])
    #m=m1+m2+m3+m4+m5+m6+m7+m8
    m=m8
    ii,jj=np.where(m>=1)
    mascara = np.zeros((m.shape[0],m.shape[1],3),dtype=np.uint8)
    mascara[ii,jj,:]=1

    ax[1].imshow(cv2.cvtColor(imagenes[i]*mascara, cv2.COLOR_BGR2RGB))
    plt.savefig('../borrar/segmentation/'+str(i)+"_m8")
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


