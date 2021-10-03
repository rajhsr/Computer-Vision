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
from google.colab.patches import cv2_imshow

# return the chi-squared distance
def chi2_distance(histA, histB, eps = 1e-9):
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
	for (a, b) in zip(histA, histB)])
	return d

def search(allfeatures , queryFeatures, limit):
    i=0
    results=[]
    result_id=[]
    for row in allfeatures:
        features=allfeatures.iloc[i,:]
        d = chi2_distance(features,queryFeatures)
        results.append(d)
        result_id.append(i)
        i=i+1
    Image_link_arr=[x for _, x in sorted(zip(results,result_id))]
    return Image_link_arr[:limit]


# google drive downloader 
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

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


# Dataset index.csv is prepared with given dataset
dataset = pd.read_csv('index.csv')
image_link = pd.DataFrame(dataset.iloc[:,0].values)
product_name = pd.DataFrame(dataset.iloc[:,1].values)
product_id = pd.DataFrame(dataset.iloc[:,2].values)
parent_category = pd.DataFrame(dataset.iloc[:,3].values)
frame_shape = pd.DataFrame(dataset.iloc[:,4].values)
Data_features = pd.DataFrame(dataset.iloc[:,5:1445].values)


# google drive source link for test image
if __name__ == "__main__":
    # https://drive.google.com/file/d/1I9soL3bS_c4p7b8rtZfaNyf6HxF3IdzN/view?usp=sharing
    # https://drive.google.com/file/d/1TXFC1NdecvBYXDuYvrL7fgy_pE6jBRWy/view?usp=sharing
    file_id = '1I9soL3bS_c4p7b8rtZfaNyf6HxF3IdzN'  ## put file id of google drive here
    destination = 'query_img.jpg'
    download_file_from_google_drive(file_id, destination)
test_img = Image.open("query_img.jpg")

cd = ColorDescriptor((8, 12, 3))

query = cv2.imread('query_img.jpg')
queryFeatures = cd.describe(query)
results = search(Data_features,queryFeatures,10)
cv2_imshow(query)
count_eyeframe=0
count_sunglasses=0
count_NonPowerReading=0
count_Rectangle=0
count_Aviator=0
count_Wayfarer=0
count_Oval=0
for i in results:
 imageID = urllib.request.urlretrieve(image_link.iloc[i,0],"dataset_img.jpg")
 if parent_category.iloc[i,0]=='eyeframe':
    count_eyeframe=count_eyeframe+1
 elif parent_category.iloc[i,0]=='sunglasses':
    count_sunglasses=count_sunglasses+1
 elif parent_category.iloc[i,0]=='Non-Power Reading':
    count_NonPowerReading=count_NonPowerReading+1

 if frame_shape.iloc[i,0]=='Rectangle':
    count_Rectangle=count_Rectangle+1
 elif frame_shape.iloc[i,0]=='Aviator':
    count_Aviator=count_Aviator+1
 elif frame_shape.iloc[i,0]=='Wayfarer':
    count_Wayfarer=count_Wayfarer+1
 elif frame_shape.iloc[i,0]=='Oval':
    count_Oval=count_Oval+1
 result = cv2.imread("dataset_img.jpg")
 cv2_imshow(result)
 cv2.waitKey(0)

maximum_parentcat=max([count_eyeframe,count_sunglasses,count_NonPowerReading])
maximum_frameshape=max([count_Rectangle,count_Aviator,count_Wayfarer,count_Oval])

if maximum_parentcat==count_eyeframe:
    print('Parent Category: eyeframe')
elif maximum_parentcat==count_sunglasses:
    print('Parent Category: sunglasses')
elif maximum_parentcat==count_NonPowerReading:
    print('Parent Category: Non-Power Reading')

if maximum_frameshape==count_Rectangle:
    print('Frame Shape: Rectangle')
elif maximum_frameshape==count_Aviator:
    print('Frame Shape: Aviator')
elif maximum_frameshape==count_Wayfarer:
    print('Frame Shape: Wayfarer')
elif maximum_frameshape==count_Oval:
    print('Frame Shape: Oval')
