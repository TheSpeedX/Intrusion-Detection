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


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def alert(number,call=False):
	cookies = {"X-App-Version": "1.0", "X-Phone-Platform": "web", "X-Default-City": "1", "X-Pincode": "400001", "XdI": "0d429faa36c459599d17506cad32cb25", "_gcl_au": "1.1.782225019.1575836626", "_omappvp": "iTEq3HaHcwk52kq9H5VOubYq7rrvfnz8pYZNWJPOeYJR14H6BOzCCODJpYMKlETqLuoAr2jH8LfGUUv7SQsToibzWk1PqWBC", "_omappvs": "1575836625625", "WZRK_S_R9Z-WWR-854Z": "%7B%22p%22%3A1%2C%22s%22%3A1575836625%2C%22t%22%3A1575836627%7D", "WZRK_G": "52d860bf981a489ca4dbd9d97078697b"}
	head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0", "Accept": "*/*", "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "x-real-ip": "", "x-ua": "", "x-ff": "", "Content-Type": "application/json", "Origin": "https://pharmeasy.in", "DNT": "1", "Connection": "close", "Referer": "https://pharmeasy.in/"}
	if call:
		r=requests.post("https://pharmeasy.in/api/auth/login", headers=burp0_headers, cookies=burp0_cookies, json={"contactNumber": number, "hasCalled": True, "profileEmail": ""}).text
	else:
		r=requests.post("https://pharmeasy.in/api/auth/requestOTP", headers=head, cookies=cookies, json={"contactNumber": number}).text
	print("Request Sent")
	return '"status":1' in r

# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# Path to frozen detection graph. This is the actual model that is used for the object detection.

#SSD_Inception Model
#PATH_TO_CKPT = 'ssd_inception_v2.pb'

#SSD MobileNet
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
FPS=10.0
RESXY=(640,480)
NOTIFY=False
PHNUM="9999999999"  #Phone Number To Notify
TIMESTAMP=datetime.now().strftime('%c').replace("/","_").replace(":","_").replace("-","_")

cap = cv2.VideoCapture(0)
if SAVE_OUTPUT:
	out = cv2.VideoWriter('output_'+TIMESTAMP+'.mp4', -1, FPS, RESXY) if SAVE_OUTPUT else None
# Running the tensorflow session
with detection_graph.as_default():
	with tf.compat.v1.Session(graph=detection_graph) as sess:
		ret = True
		try:
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
						out.write(img)
						NOTIFY=not alert(PHNUM) if NOTIFY else False
				cv2.waitKey(1)
		except:
			if SAVE_OUTPUT:
				out.release()
			cap.release()
			cv2.destroyAllWindows()