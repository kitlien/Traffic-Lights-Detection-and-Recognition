import cv2
import os

video_path = 'test_video.mp4'
camera = cv2.VideoCapture(video_path)

ii = 0
while True:
	res, image = camera.read()
	if image is None:
		print 'empty'
	ii = ii + 1
	name = 'demo_images/%06d.jpg'%(ii)
	cv2.imwrite(name, image)
	#cv2.imshow('test', image)
	#cv2.waitKey(100)

