import numpy as np
import pandas as pd
import csv
import os
import urllib.request
from PIL import Image
import requests
import cv2
import imutils
import argparse
import glob

# Dataset
dataset = pd.read_csv('eyewear_ml_challenge.csv')
#dataset=dataset.drop(dataset.index[185])
product_name = pd.DataFrame(dataset.iloc[:,1].values)
product_id = pd.DataFrame(dataset.iloc[:,2].values)
parent_category = pd.DataFrame(dataset.iloc[:,3].values)
Image_Front = pd.DataFrame(dataset.iloc[:,4].values)
frame_shape = pd.DataFrame(dataset.iloc[:,5].values)

## color descriptor 
class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins
	def describe(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
  
	# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]
		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# extract a color histogram from the image, then update the
			# feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)
		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		# normalize the histogram if we are using OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		# otherwise handle for OpenCV 3+
		else:
			hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist


cd = ColorDescriptor((8, 12, 3))

i=0
n=len(Image_Front.index);
output = open('index1.csv', "w") # write the dataset file
n_final=0;
while i<n :
		url=Image_Front.iloc[i,0]
		productname=product_name.iloc[i,0]
		productid=product_id.iloc[i,0]
		parentcategory=parent_category.iloc[i,0]
		frameshape=frame_shape.iloc[i,0]
		response = requests.get(url)
		if response.status_code == 200 :
			imageID = urllib.request.urlretrieve(url,"dataset_img.jpg")
			image = cv2.imread('dataset_img.jpg')
			features = cd.describe(image)
			features = [str(f) for f in features]
			output.write("%s,%s,%s,%s,%s,%s\n" % (url,productname,productid,parentcategory,frameshape, ",".join(features)))
			n_final=n_final+1

		i=i+1
 
# close the index file
output.close()

  
