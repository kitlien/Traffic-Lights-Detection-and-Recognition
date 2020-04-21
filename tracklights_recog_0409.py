#coding=utf-8
# 加载库
import numpy as np
import os
import requests
import argparse
import time
import json
import sqlite3
import cv2
import sys
import logzero
import logging
from logzero import setup_logger
from skimage import segmentation,measure
import matplotlib.patches as mpatches
import datetime
import platform
import math
import glob
reload(sys)  
sys.setdefaultencoding('utf8')   
#========================== 配置参数 ===============================
formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(module)s: %(lineno)d - %(levelname)s: %(message)s')
log = setup_logger(name="tracklights", logfile="tracklights_recog.log", formatter=formatter)

parser = argparse.ArgumentParser()
parser.add_argument('-queryImgURL', help="image list query url", type=str, default="http://121.36.142.163:8090/warn/json/getXinhaodengPic")
parser.add_argument('-uploadRecogURL', help="image recog result upload", type=str, default="http://121.36.142.163:8090/warn/json/xhdProcessCallback")
parser.add_argument('-savePATH', help="save path.", type=str, default = os.getcwd())
args = parser.parse_args()

#图像算法参数
dst_img_size = (960, 540)
kernelsize = (7, 7)
minArea = 800
maxArea = 40000
ratio_thr = 0.7
debug = False
PI = 3.1415926
database = "tracklights_result.db"
recog_result = {0: "no light", 1: "green light",2: "yellow light", 3: "red light", 4: "blue light", 5: "white light",}

log.info("start...")
log.info("Args: %s", args)
# ==============================

def get_img_list(queryUrl):
	'''获取图像列表并返回。'''
	response = requests.get(queryUrl)
	if response.status_code ==200:
		response = response.json()
	else:
		log.error("Can't get the video list...")
		return None
	if response['responseCode'] == '0000':
		log.info("got the img list.")
		img_list = response['info'].split(',')
		return img_list
	else:
		return None

def find_lights(cimg, cont_img, color_name):
	has_lights = 0
	font = cv2.FONT_HERSHEY_SIMPLEX   #initialize font
	segmentation.clear_border(cont_img)  #eliminate object adjacency to boarder
	label_image = measure.label(cont_img)  #label connected area
	borders = np.logical_xor(cont_img, cont_img) #Xor
	label_image[borders] = -1
	#image_label_overlay_g = color.label2rgb(label_image_g, image=opened_g) #
	x,y = 1,1
	radius = 1
	for region in measure.regionprops(label_image): #enumerate every connected area
		if region.convex_area < 300 or region.area > 4000:
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

		if eccentricity <= 0.4 or circularity >= 0.6 or circum_circularity >= 0.6:
			#cv2.circle(cimg, (y,x), radius, (0,255,0),3)
			#cv2.putText(cimg,color_name,(y,x), font, 1,(0,255,0),2)
			has_lights = 1
			return has_lights,y,x,radius
	return has_lights, y, x, radius

def find_lights_yellow(cimg, cont_img, opened_r, color_name):
	font = cv2.FONT_HERSHEY_SIMPLEX   #initialize font
	segmentation.clear_border(cont_img)  #eliminate object adjacency to boarder
	label_image = measure.label(cont_img)  #label connected area
	borders = np.logical_xor(cont_img, cont_img) #Xor
	label_image[borders] = -1
	mask_img = np.zeros((opened_r.shape[0],opened_r.shape[1]), dtype = np.uint8)
	x,y,r = 1,1,1
	light_state = 0
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

		if eccentricity <= 0.4 or circularity >= 0.6 or circum_circularity >= 0.6:
			#may be yellow color in a red light, make a larger mask, recompute the light color
			cv2.circle(mask_img, (y,x), radius+int(radius*0.5), (255,255,255),-1)
			ex_color_area_r = cv2.bitwise_and(opened_r, mask_img)
			
			light_state,y_r,x_r,r_r = find_lights(cimg, ex_color_area_r, 'red')
			if light_state == 1:
				# is red color
				light_state = 3
				return light_state,y_r,x_r,r_r
			else:
				# is yellow color
				light_state = 2
				return light_state,y,x,radius

	return light_state, y, x, r
	
def recog_light_color_demo(img_name):
	'''0无信号灯,1绿灯,2黄灯,3红灯,4蓝灯,5白灯'''
	# read image
	if platform.system() == "Windows":
		img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), -1)
	else:
		#img = cv2.imread(img_name)
		img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), -1)

	img = cv2.resize(img, dst_img_size)
	img = img[100 : -1, np.int(img.shape[1]/2) :, :]
	cimg = img
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#BGR2HSV

	# color range
	lower_red1 = np.array([0,80,80])		#define the red channel range in HSV space
	upper_red1 = np.array([10,255,255])
	lower_red2 = np.array([160,80,80])	  #define the red channel range in HSV space
	upper_red2 = np.array([180,255,255])
	lower_green = np.array([40,50,50])		#define the green channel range in HSV space
	upper_green = np.array([90,255,255])
	lower_yellow = np.array([10,80,80])	 #define the blue channel range in HSV space
	upper_yellow = np.array([35,255,255])
	lower_white = np.array([0,0,221])	 #define the blue channel range in HSV space
	upper_white = np.array([180,80,255])
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
	light_state = -1
	y,x,r = 1,1,1
	for region_w in measure.regionprops(label_image_w): #enumerate every connected aread   
		#ignore small and large area	
		if region_w.convex_area < 500 or region_w.area > 2000:
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
			circularity = 4*PI*area/(perimeter*perimeter)
			circum_circularity = 4*PI*convex_area/(4*PI*PI*radius*radius) 
		
		if eccentricity <= 0.4 or (circularity >= 0.7 or circum_circularity >= 0.73):
			cv2.circle(mask_img, (y,x), radius+int(radius*0.5), (255,255,255),-1)

			ex_color_area_r = cv2.bitwise_and(opened_r, mask_img)
			ex_color_area_g = cv2.bitwise_and(opened_g, mask_img)
			ex_color_area_y = cv2.bitwise_and(opened_y, mask_img)

			is_r, y_r,x_r,r_r = find_lights(cimg, ex_color_area_r, 'red')
			is_g, y_g,x_g,r_g = find_lights(cimg, ex_color_area_g, 'green')
			is_y, y_y,x_y,r_y = find_lights_yellow(cimg, ex_color_area_y, opened_r, 'yellow') #maybe yellow color in red light

			if is_r:
				light_state = 3
				return light_state, y_r,x_r,r_r
			if is_g:
				light_state = 1
				return light_state, y_g,x_g,r_g
			if is_y>0:
				light_state = is_y
				return light_state,y_y,x_y,r_y
		else:
			continue

	is_r,y_r,x_r,r_r = find_lights(cimg, opened_r, 'red')
	is_g, y_g,x_g,r_g = find_lights(cimg, opened_g, 'green')	
	is_y, y,x,r = find_lights(cimg, opened_y, 'yellow')
	if is_r:
		light_state = 1
		return light_state, y_r,x_r,r_r
	if is_g:
		light_state = 3
		return light_state, y_g,x_g,r_g
	if is_y>0:
		light_state = 2
		return light_state,y,x,r 

	return 0,y,x,r

def send_status(img_name, recog_id, sendURL):
	sendStr = img_name + ":" + str(recog_id)	   
	data = {'warns': sendStr}
	print(data)
	log.info(data)
	try:
		response = requests.post(sendURL, json=data)
		if response.status_code ==200:
			response = response.json()
		else:
			log.error("Can't upload the result...")
			return False	 
		
	except Exception as e:
		log.exception(e)
		return False
		
	if response['responseMsg'] == 'success':
		return True
	else:
		log.error(response['responseMsg'])
		return False
 
def update_database(db_name, img_name, recog_result_id, recog_result):
	color =  recog_result[recog_result_id]
	conn = sqlite3.connect(db_name)
	c = conn.cursor()
	sql = 'INSERT INTO tracklights_recog_result (img_id, img_name, recog_result, result_id) VALUES (NULL, "{}","{}", {})'.format(img_name, color, recog_result_id)
	
	# sql = 'INSERT INTO tracklights_recog_result (img_id, img_name, recog_result, result_id) VALUES ' +'"' + '({},{},{},{})'.format('NULL', img_name, color, 0) + '"'
	c.execute(sql)
	conn.commit() 
	conn.close()
		 
if __name__ =="__main__":
	# update_database(database, "test.png", 5, recog_result)
	'''
	save_name = '/home/inspur/lianjie/Traffic-Lights-Detection-and-Recognition/samples/000063.jpg'
	recog_result_id,y,x,r = recog_light_color_demo(save_name)
	print 'result', recog_result_id, y, x, r
	'''
	while True:
		try:
			folderName = "{}-{}-{}".format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day)			
			folderName = os.path.join(args.savePATH, folderName)
			isExists=os.path.exists(folderName)
			if not isExists:
				os.makedirs(folderName)
			img_list = get_img_list(args.queryImgURL)
			
			if img_list is not None:
				for i in img_list:
					print(i)
			else:
				log.info("no img to be processed...")
				
			if img_list is not None:
				for img_name in img_list:
					# 步骤1，下载图像
					_, save_name_relative = os.path.split(img_name)
					response = requests.get(img_name)					
					save_name = os.path.join(folderName, save_name_relative)
					video_bytes = response.content
					fid = open(save_name, 'wb')
					fid.write(video_bytes) 
					fid.close()
					log.info(img_name + " has downloaded.")
					time.sleep(2)		 
					# img_name = 'C:\\Users\\liyhe\\Desktop\\traffic_light_classified-master\\1 (27).png'
					
					# 步骤,2, 识别图像红绿灯颜色 
					recog_result_id,_,_,_ = recog_light_color_demo(save_name)
					log.info("{} has processed.".format(save_name))
					
					
					result = send_status(img_name, recog_result_id, args.uploadRecogURL)
					log.info("send process result to server: %s", result)
					
					update_database(database, save_name, recog_result_id, recog_result)
			else:
				log.info("no video. wait for 10 seconds.")
				time.sleep(10)
		except Exception as e:
			log.exception(e)
			time.sleep(100)
	


			

  
	
