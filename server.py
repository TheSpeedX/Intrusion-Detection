from flask import Flask, request, Response
import numpy as np
import cv2
import os,time
import json

app = Flask(__name__)


@app.route('/save/<cam_id>', methods=['POST'])
def test(cam_id):
	r = request
	# convert string of image data to uint8
	nparr = np.frombuffer(r.data, np.uint8)
	# decode image
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	if not os.path.exists(cam_id):
		os.mkdir(cam_id)
	cv2.imwrite(cam_id+"/"+cam_id+time.strftime("%Y%m%d_%H%M%S")+".jpg",img)
	# do some fancy processing here....

	# build a response dict to send back to client
	response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}

	return Response(response=json.dumps(response), status=200, mimetype="application/json")


# start flask app
app.run(host="localhost", port=5000)