import requests
from flask import Flask, render_template, redirect, url_for, flash, jsonify
# from flask_bootstrap import Bootstrap
from flask_restplus import reqparse


from flask import Flask
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from test import *
import numpy as np

app = Flask(__name__)
api = Api(app, default = "comp9321-Project3",title= "Wine Price prediction", description="Our api is used to provide functions of wine")

argument = {1:'age',2:'sex',3:'cpt',4:'rbp',5:'sc',6:'fbs',7:'rer',8:'mhr',9:'eia',10:'op',11:'sts',12:'mv',13:'thal'}
fake_factors = get_important_factors_list() # from machine learning 

@app.route('/main/value',methods=['POST'])
@api.response(200, 'OK')
@api.response(400, 'Validation Error')
@api.doc(description="Depending on input info to predict the price and make recommendation")
def value():
    parser = reqparse.RequestParser()
    for i in range(len(fake_factors)):
    	parser.add_argument(argument[fake_factors[i]], type=str)
    args = parser.parse_args()
    attrs = []
    for i in range(len(fake_factors)):
    	attrs.append(float(args.get(argument[fake_factors[i]])))
    print(attrs)
    return jsonify(predict_user_input(np.array(attrs))), 200



if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=3000)


    '''
    <form action="/main" class="back_to_home" id="back" method="get">
        <input type="submit" value="Back">
    </form>
    '''