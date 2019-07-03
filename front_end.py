import requests,os
from flask import Flask, render_template, redirect, url_for,flash
from flask_bootstrap import Bootstrap
from flask_restplus import reqparse
from markupsafe import Markup
from visualise import *
from test import *

if not os.path.exists("pic"):
    os.mkdir("pic")

app = Flask(__name__)
# print(app.config['UPLOAD_FOLDER'])

argument = {1:'age',2:'sex',3:'cpt',4:'rbp',5:'sc',6:'fbs',7:'rer',8:'mhr',9:'eia',10:'op',11:'sts',12:'mv',13:'thal'}
names = ["age","sex","chest_pain_type","resting_blood_pressure","serum_cholestoral",\
        "fasting_blood_sugar","resting_electrocardiographic_results","maximum_heart_rate_achieved","exercise_induced_angina",\
        "oldpeak","the_slope_of_the_peak_exercise_ST_segment","number_of_major_vessels_colored_by_ourosopy","thal","target"]
# fake_factors = [1,2,3,4,5,8,9,10,13] # from machine learning 
fake_factors = get_important_factors_list()
temp = predict_model()

def makeup_inputbox(important_factors):
    abbr = []
    for i in important_factors:
        abbr.append([argument[i],names[i-1]])
    return abbr

@app.route('/',methods=['GET'])
def homepage():
    return render_template('home.html')

@app.route('/main',methods=['GET'])
def visulization():
    # print(os.listdir('pic/'))
    pics = os.listdir('static/pic/')
    # print(pics)
    # pic = makeup_pics(pics)
    # pic1 = 'static/pic' +  '/chest_pain_type.png'
    # pic2 = 'static/pic' +  '/exercise_induced_angina.png'
    # print(pic)
    
    return render_template('main.html',results = pics)

@app.route('/main/prediction',methods=['GET'])
def prediction():
    # predict_model()
    abbr = makeup_inputbox(fake_factors)
    pics = os.listdir('static/predict/')
    designed_list = ['check_imbalance.png','important_factors.png','iterations_vs_accuracy.png','iterations_vs_accuracy_cv.png','confusion_matrix.png','roc.png']
    if set(pics) == set(designed_list):
        return render_template('prediction.html',result = designed_list,results = abbr,kfold = temp[-1])
    else:
        return render_template('prediction.html',result = pics,results = abbr,kfold = temp[-1])

@app.route('/main/prediction_result',methods=['GET'])
def prediction_result():
    parser = reqparse.RequestParser()
    print(parser,'\n')

    for i in range(len(fake_factors)):
    	parser.add_argument(argument[fake_factors[i]], type=str)
    args = parser.parse_args()
    print('\nargs:',args,'543435354353543')

    attrs = []
    for i in range(len(fake_factors)):
    	attrs.append(args.get(argument[fake_factors[i]]))
    print('\nattrs:',attrs)

    url = 'http://127.0.0.1:3000/main/value?' + argument[fake_factors[0]] + '=' + attrs[0]
    
    for i in range(1,len(fake_factors)):
    	url += '&'+ argument[fake_factors[i]] +'='+ attrs[i]
    print('\nurl:',url)

    response = requests.post(url, headers={"Accept": "application/json"})
    data = response.json()
    # return render_template('prediction_result.html', price=price,table=Markup(table))
    return render_template('prediction_result.html', pred=data)

if __name__ == '__main__':
    if not os.path.exists("static/pic"):
        os.mkdir("static/pic")
    if not os.path.exists("static/predict/"):
        os.mkdir("static/predict/")     
    if len(os.listdir('static/pic/')) == 0:
        visualisation()
    app.debug = True
    app.run(host='127.0.0.1', port=5000)

