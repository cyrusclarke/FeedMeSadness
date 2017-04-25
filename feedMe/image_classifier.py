import numpy as np
from flask import Flask, abort, jsonify, render_template, request
import cPickle as pickle 
import vgg16
from vgg16 import Vgg16


my_data = pickle.load(open("data.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
	#get JSON from the post
	data = request.get_json(force=True)
	#convert our JSON to a numpy array
	predict_request = [data['v1']]
	#put the data into a np array
	predict_request = np.array[predict_request]
	#run the np array through the pickled data
	y_hat = my_data.predict(predict_request)

	#return predicion (only 1)
	output = [y_hat(0)]
	#take the list and convert to JSON
	return jsonify(results=output)




if __name__ == '__main__':
    app.run()