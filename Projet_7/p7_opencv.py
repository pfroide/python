#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:01:18 2018

@author: toni
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

dataset_path = '/home/toni/Bureau/p7/Images/n02085620-Chihuahua'

def image_detect_and_compute(detector, img_name):
    """
    Detect and compute interest points and their descriptors.
    """

    img = cv2.imread(os.path.join(dataset_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des

liste_dossier = '/home/toni/Bureau/p7/Images/n02085620-Chihuahua'
orb = cv2.ORB_create()

detector = orb
key_points = []
description = []

for i, filename in enumerate(os.listdir(liste_dossier)):
    print(i)
    # On ne garde que NB_EXEMPLES exemplaires de chaque race
    a, b, c = image_detect_and_compute(detector, filename)
    key_points.append(b)
    description.append(b)
