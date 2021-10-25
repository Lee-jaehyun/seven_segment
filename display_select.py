from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import optimizers
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from skimage.transform import resize
import time
import datetime


### load model
model1 = load_model('display_classification_VGG1022_0001_224_300.h5')

def select(test_images):
    print(len(test_images))
    
    for i in range(len(test_images)):
        test_image = test_images[i]

        test_img = cv2.resize(test_image, dsize=(300, 300), interpolation=cv2.INTER_AREA)

        test_num = np.asarray(test_img, dtype=np.float32)
        test_num = np.expand_dims(test_num, axis=0)
        test_num = test_num / 255

        pred = model1.predict(test_num).reshape(test_num.shape[0])
        print("결과값 :", pred)
        if pred >= 0.6:
            output = 1
        else:
            output = 0
        
        print("디스플레이 영역 결과 :", output)

        if output == 0:
            cv2.imshow("test!!!!!!", test_img)
            cv2.waitKey(0)
            return test_image
        
     
        else:
            continue

    return "None"
    
    
    
        



