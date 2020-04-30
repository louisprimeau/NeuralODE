from time import time
import os
import numpy as np
from PIL import Image

def data(path):
    start = time() # Timing function
    images, groundtruths = [], []
    for file in os.listdir(path): #Iterate through train folder
        images.append(np.asarray(Image.open(path + file)).reshape(-1,1) / 255) #Load image into memory
        groundtruths.append(np.asarray((int(file[10]))*[0] + [1] + (9 - int(file[10]))*[0])) #literally do not ever do this
        #break #REMOVE LATER, ONLY LOADS ONE IMAGE
    print("Data Loaded... ", len(images), "images loaded, ", round(time() - start, 2), "s elapsed.")
    return(images, groundtruths)
