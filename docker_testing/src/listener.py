from flask import Flask
from flask import request
from flask_restful import Resource, Api, reqparse

import pandas as pd
import ast
import os
from dotenv import load_dotenv
from datetime import date
import download
import json

load_dotenv()

PORT = os.getenv('PORT');

app = Flask(__name__)
api = Api(app)

class Test(Resource):
    def get(self):
        data = {
            "test": "isWorking",
            "ok?": "ok"
        }
        return {'data' : data}, 200

class Images(Resource):
    data = pd.DataFrame()
    def post(self):
        json_data = request.get_json()
        coordinates = json_data['coordinates']
        time_start = json_data['time_start']
        time_end = json_data['time_end']
        cloud_range = json_data['cloud_range']
        

        # new_data = pd.DataFrame({
        #     'coordinates': json_data['coordinates'],
        #     'time_start': json_data['time_start'],
        #     'time_end': json_data['time_end'],
        #     'cloud_range': json_data['cloud_range'],
        # })
        download.download_images(coordinates, date(time_start[0],time_start[1],time_start[2]), date(time_end[0],time_end[1],time_end[2]), (0, cloud_range))
        # Images.data = pd.concat([Images.data, new_data], ignore_index=True)
        return {'Images': "downloaded"}, 200

api.add_resource(Test, '/test')
api.add_resource(Images, '/images')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=PORT) 