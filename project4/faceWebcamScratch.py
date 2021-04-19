import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pca_cov

captureMac = cv2.VideoCapture(0)

frame_list = []
while True:
    ret,frame = captureMac.read()
    print(frame)
    frame_list.append(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
captureMac.release()
try:
    cv2.destroyWindow()
except:
    print(f'window destroyed')
print('DONE')