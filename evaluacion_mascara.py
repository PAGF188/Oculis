import cv2
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Segmentation evaluation. Must be in same order')
parser.add_argument('-r','--real', help='<Required> Place real segmentations', required=True)
parser.add_argument('-p','--predict', help='<Required> Place predicted segmentations', required=True)
args = parser.parse_args()

sumatorio = 0
im=0
if os.path.isdir(args.real) and os.path.isdir(args.predict):
    for real_n,predicha_n in zip(os.listdir(args.real),os.listdir(args.predict)):
        real = cv2.imread(os.path.join(args.real,real_n))
        real_ = real[:,:,0]*0
        ix,iy = np.where(real[:,:,0]==255)
        real_[ix,iy]=1

        predicha = cv2.imread(os.path.join(args.predict,predicha_n))
        predicha_= predicha[:,:,0]*0
        ix,iy = np.where(predicha[:,:,0]>0)
        predicha_[ix,iy]=1
        
        overlap = len(np.where(real_ * predicha_==1)[0])
        union =  len(np.where(np.logical_or(real_,predicha_)==1)[0])
        sumatorio += overlap/union
        im+=1

print("IoU:",sumatorio/im)