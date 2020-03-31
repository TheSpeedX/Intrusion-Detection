import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from datetime import datetime
import requests
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import json
import time
import threading


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



CAMID="CAM001"
APIURL="http://localhost:5000/save/"+CAMID

def send_frame(frame_queue):
	c=0
	while True:
		print(len(frame_queue))
		while len(frame_queue)==0:
			c+=1
			print("sleeped ",c)			
			time.sleep(1)		
		frame=frame_queue.pop(0)
		_, img_encoded = cv2.imencode('.jpg', frame)
		response = requests.post(APIURL, data=img_encoded.tostring(), headers= {'content-type': 'image/jpeg'})
		print(response.text)




PATH_TO_CKPT = 'ssd_mobilenet.pb'


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'


NUM_CLASSES = 90

tf.gfile = tf.io.gfile
# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def =	tf.compat.v1.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.	 Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

SAVE_OUTPUT=True
MIN_DETECT=0.4
RESXY=(640,480)
NOTIFY=False
PHNUM="9999999999"  #Phone Number To Notify
TIMESTAMP=datetime.now().strftime('%c').replace("/","_").replace(":","_").replace("-","_")



def main(frame_queue):
	cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
	with detection_graph.as_default():
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			ret = True
			# try:
			while (ret):
				ret,image_np = cap.read()
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
				  image_np,
				  np.squeeze(boxes),
				  np.squeeze(classes).astype(np.int32),
				  np.squeeze(scores),
				  category_index,
				  use_normalized_coordinates=True,
				  line_thickness=8)
				img=cv2.resize(image_np,RESXY)
				cv2.imshow('Human_Detect',img)
				if SAVE_OUTPUT:
					found=np.nonzero(scores>=MIN_DETECT)
					if 1 in classes[found]:
						cv2.putText(img,datetime.now().strftime('%c'),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
						#out.write(img)
						frame_queue.append(img)
						#NOTIFY=not alert(PHNUM) if NOTIFY else False
				cv2.waitKey(10)
			# except:
				# cap.release()
				# cv2.destroyAllWindows()
if __name__=='__main__':

	frame_queue = []
	p1=threading.Thread(target=main, args=(frame_queue,))
	p2=threading.Thread(target=send_frame,args=(frame_queue,))
	p2.daemon=True
	p1.start()
	time.sleep(5)
	p2.start()
	p1.join()
	p2.join()
	print("Job Completed")
