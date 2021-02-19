
from keras.models import load_model
import numpy as np
import cv2 as cv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import os
from imutils import paths
import glob
import numpy as np
import cv2 as cv
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, default="Data/video.mp4", help="path to video")
ap.add_argument("-m", "--model", type=str, default="model/model2.h5", help="path to trained cartoon character recognition model")
args = vars(ap.parse_args()) 

class_dict ={ 0 : "Mickey", 1: "Pooh", 2 : "Donald", 3 : "Minion"}
window_name = 'Cartoon Character Recognition'
font = cv.FONT_HERSHEY_DUPLEX
org = (10, 50) 
fontScale = 1
color = (0, 0, 0) 
thickness = 2


model = load_model(args['model'])
cap = cv.VideoCapture(args['path'])

while(cap.isOpened()):
    ret, frame = cap.read()    
    frame = cv.resize(frame, (700, 650), interpolation = cv.INTER_CUBIC)
    img = cv.resize(frame, (256, 256), interpolation = cv.INTER_CUBIC)
    n = img
    n = n.reshape(1, 3, 256, 256).astype('float32')
    cnn_probab = model.predict(n)
    predict = np.argmax(cnn_probab)
    
    frame = cv.putText(frame, class_dict[predict], org, font,  
                       fontScale, color, thickness, cv.LINE_AA)
    print(class_dict[predict])
    
    smallImg=cv.imread('Data/'+class_dict[predict]+'.jpeg')
    frame[60:60+smallImg.shape[0], 10:10+smallImg.shape[1]] = smallImg

    # Displaying the image 
    cv.imshow(window_name, frame)
    if cv.waitKey(1) & 0xFF == ord('s'):
        start = True
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



