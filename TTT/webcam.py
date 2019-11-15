import urllib.request
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import io

def get_image_from_device(ip, imsize = (512, 384)):

	URL = "http://{}/photo.jpg".format(ip)
	img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
	#image = io.imread(URL, plugin='matplotlib')
	#print(image)
	img = cv2.resize(cv2.cvtColor(cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), imsize)
	
	#cv2.imshow('IPWebcam',img)

	return((img/255.).astype(np.float32))
		

if __name__ ==  '__main__':	


	plt.imshow(get_image_from_device())
	plt.show()