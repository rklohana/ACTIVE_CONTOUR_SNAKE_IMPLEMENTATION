import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
def line_energy(image):


    return image #(image-image.min())/(image.max()-image.min())
def edge_energy(image):
    gx=cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    gy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1)
    C=((gx**2)+(gy**2))**0.5
    return C

def term_energy(image):
    gx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, dx=1, dy=0)
    gxy = cv2.Sobel(gx, cv2.CV_64F, dx=1, dy=0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, dx=0, dy=1)
    E_term=((gxx*(gy**2))-(2*gxy*gx*gy)+(gyy*(gx**2)))/((((gx**2)+(gy**2))**(1.5))+1)
    return E_term



def external_energy(image, w_line, w_edge, w_term):
    e_edge = edge_energy(image)
    e_term=term_energy(image)
    e_line=line_energy(image)
    e_edge=edge_energy(image)
    e_ext=(e_term*w_term)+(e_edge*w_edge)+(e_line*w_line)

    cv2.imshow('ext',e_ext)
    cv2.waitKey(0)
    return e_ext
