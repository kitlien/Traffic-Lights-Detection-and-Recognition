import os
import cv2
import cv2.cv as cv
import sys
import socket
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import segmentation,measure

DEBUG = True

HOST = '192.168.1.166'
PORT = 10006
strLight = ""
dstSize = (960, 540)
PI = 3.1415926

def find_lights(cimg, cont_img, color_name):
	font = cv2.FONT_HERSHEY_SIMPLEX   #initialize font
	segmentation.clear_border(cont_img)  #eliminate object adjacency to boarder
	label_image = measure.label(cont_img)  #label connected area
	borders = np.logical_xor(cont_img, cont_img) #Xor
	label_image[borders] = -1
	#image_label_overlay_g = color.label2rgb(label_image_g, image=opened_g) #
	for region in measure.regionprops(label_image): #enumerate every connected area
		if region.convex_area < 130 or region.area > 2000:
			continue
		area = region.area   #area of connected region
		eccentricity = region.eccentricity   #eccentric
		convex_area  = region.convex_area	#convex area
		minr, minc, maxr, maxc = region.bbox #
		radius	   = max(maxr-minr,maxc-minc)/2	#longer axis of connected area
		centroid	 = region.centroid		   #barycenter coordinate
		perimeter	= region.perimeter	  #perimeter of the area
		x = int(centroid[0])
		y = int(centroid[1])
		rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								  fill=False, edgecolor='red', linewidth=2)

		if perimeter == 0:
			circularity = 1
		else:
			circularity = 4*PI*area/(perimeter*perimeter)
			circum_circularity	  = 4*PI*convex_area/(4*PI*PI*radius*radius) 

		if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.8:
			cv2.circle(cimg, (y,x), radius, (0,255,0),3)
			cv2.putText(cimg,color_name,(y,x), font, 1,(0,255,0),2)
			return color_name,cimg
		else:
			continue

	
def detect(img):
	font = cv2.FONT_HERSHEY_SIMPLEX   #initialize font
	#img = cv2.imread(filepath + '/' + file)

	img = cv2.resize(img, dstSize)
	img = img[100 : -50, np.int(img.shape[1]/2) :, :]
	cimg = img
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#BGR2HSV

	# color range
	lower_red1 = np.array([0,80,80])		#define the red channel range in HSV space
	upper_red1 = np.array([10,255,255])
	lower_red2 = np.array([160,80,80])	  #define the red channel range in HSV space
	upper_red2 = np.array([180,255,255])
	lower_green = np.array([40,50,50])		#define the green channel range in HSV space
	upper_green = np.array([90,255,255])
	lower_yellow = np.array([10,43,46])	 #define the blue channel range in HSV space
	upper_yellow = np.array([35,255,255])
	lower_white = np.array([0,0,221])	 #define the blue channel range in HSV space
	upper_white = np.array([180,30,255])
	#get each traffic lights part according to color space definition
	mask1 = cv2.inRange(hsv, lower_red1,   upper_red1)
	mask2 = cv2.inRange(hsv, lower_red2,   upper_red2)
	mask_g = cv2.inRange(hsv, lower_green,  upper_green)
	mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
	mask_r = cv2.add(mask1, mask2)
	mask_w = cv2.inRange(hsv, lower_white, upper_white)

	#define struct element, morphology open operation
	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))#MORPH_CROSS is better than EMORPH_RECT
	#open operation
	opened_r  = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, element)
	opened_g  = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, element)
	opened_y  = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, element)
	opened_w  = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, element)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))#MORPH_CROSS is better than EMORPH_RECT
	opened_w  = cv2.morphologyEx(opened_w, cv2.MORPH_CLOSE, element)
	
	################detect white lighsts area (often in the center of lights)########################
	segmentation.clear_border(opened_w)  #eliminate object adjacency to boarder
	label_image_w = measure.label(opened_w)  #label connected area
	borders_w = np.logical_xor(mask_w, opened_w) #xor
	label_image_w[borders_w] = -1

	mask_img = np.zeros((opened_w.shape[0],opened_w.shape[1]), dtype = np.uint8)

	for region_w in measure.regionprops(label_image_w): #enumerate every connected aread   
		#ignore small and large area	
		if region_w.convex_area < 120 or region_w.area > 2000:
			continue
		#draw bounding box
		area = region_w.area				   #connected area, pixel nums in area
		eccentricity = region_w.eccentricity   #eccentric
		convex_area  = region_w.convex_area	#convex area
		minr, minc, maxr, maxc = region_w.bbox #
		radius = max(maxr-minr,maxc-minc)/2	#longer axis of connected area
		centroid = region_w.centroid		   #barycenter coordinate
		perimeter	= region_w.perimeter	  #perimeter of area
		x = int(centroid[0])
		y = int(centroid[1])
		rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								  fill=False, edgecolor='red', linewidth=2)

		if perimeter == 0:
			circularity = 1
		else:
			circularity = 4*3.141592*area/(perimeter*perimeter)
			circum_circularity = 4*3.141592*convex_area/(4*3.141592*3.141592*radius*radius) 

		if eccentricity <= 0.4 or circularity >= 0.7 or circum_circularity >= 0.73:
			#cv2.circle(cimg, (y,x), radius, (0,0,255),3)
			#cv2.putText(cimg,'RED',(y,x), font, 1,(0,0,255),2)
			cv2.circle(mask_img, (y,x), radius+int(radius*0.5), (255,255,255),-1)

			ex_color_area_r = cv2.bitwise_and(opened_r, mask_img)
			ex_color_area_g = cv2.bitwise_and(opened_g, mask_img)
			ex_color_area_y = cv2.bitwise_and(opened_y, mask_img)
			find_lights(cimg, ex_color_area_r, 'red')
			find_lights(cimg, ex_color_area_g, 'green')			
			find_lights(cimg, ex_color_area_y, 'yellow')

			return "RED",cimg
		else:
			continue

	find_lights(cimg, opened_r, 'red')
	find_lights(cimg, opened_g, 'green')			
	find_lights(cimg, opened_y, 'yellow')

	return "NONE",img

def max(a, b):
	if a>b:
		return a
	else: 
		return b

if __name__ == '__main__':
	path = '/home/inspur/lianjie/stop_line_samples'
	ii = 0
	videoWriter = cv2.VideoWriter("save.avi", cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 20.0, (1200,800))
	video_path = 'test_video.mp4'
	camera = cv2.VideoCapture(video_path)
	while True:
	#for line in os.listdir(path):
		#filename = os.path.join(path, line)
		res, image = camera.read()
		#image = cv2.imread(filename)
		strLight,result = detect(image)
		ii = ii + 1
		savename = 'demo_result/%06d.jpg'%(ii)
		cv2.imwrite(savename, result)
		#cv2.imshow("result", result)
		#cv2.waitKey(100)
		videoWriter.write(result) #write video frame

	#del(capture) 
