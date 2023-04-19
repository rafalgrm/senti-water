from flask import Flask
from flask_restful import Resource, Api, reqparse
import ast
import os
from dotenv import load_dotenv

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


api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=PORT) 