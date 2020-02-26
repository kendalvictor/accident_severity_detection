#!/usr/bin/env python
# coding: utf-8

"""
    Car damage classification Api
    ============================================
    Free Poc:

    Note: 
    Structure:
        model/
        resources/
        app.py
        classifier.py
    _copyright_ = 'Copyright (c) 2019 J.W. - Everis', 
    _license_ = GNU General Public License
"""

# Load libraries
import base64

from flask import Flask, request, json 
from flask_restful import Resource, Api

from classifier import CarDamageClassification

app = Flask(__name__)
api = Api(app)

model = CarDamageClassification('models/')

class CarDamage(Resource):
    def get(self):
        return {'response': 'Use post method'}
    
    def post(self):
        if request.headers['Content-Type'] == 'application/image':
            req = request.data
            image_decoded = base64.b64decode(req)
            #print(type(image_decoded))
            response = model.predict(image_decoded)
            
            return response

        else:
            return {'response': 'Incorrect header'}


api.add_resource(CarDamage, '/predict')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7000)
