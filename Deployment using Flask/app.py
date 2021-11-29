import os
from flask import Flask, request 
from flask.templating import render_template
import sklearn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import keras 
from keras.applications.vgg16 import VGG16
import cv2
import pickle
size = 224


svm1 = pickle.load(open('svm1.sav','rb'))
svm2 = pickle.load(open('svm2.sav','rb'))
svm3 = pickle.load(open('svm3.sav','rb'))
svm4 = pickle.load(open('svm4.sav','rb'))
svm5 = pickle.load(open('svm5.sav','rb'))
svm6 = pickle.load(open('svm6.sav','rb'))
#svm7 = pickle.load(open('svm7.sav','rb'))
#svm8 = pickle.load(open('svm8.sav','rb'))
#svm9 = pickle.load(open('svm9.sav','rb'))
#svm10 = pickle.load(open('svm10.sav','rb'))
#svm11 = pickle.load(open('svm11.sav','rb'))
#svm12 = pickle.load(open('svm12.sav','rb'))


#models  = [svm1,svm2,svm3,svm4,svm5,svm6,svm7,svm8,svm9,svm10,svm11,svm12]
models  = [svm1,svm2,svm3,svm4,svm5,svm6]


vgg_model = VGG16(weights= 'imagenet', include_top=False, input_shape=(size,size,3))

def new_sample_feature(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image,(size,size))
    image = np.array([image])
    feature_extractor = vgg_model.predict(image)
    features = feature_extractor.reshape(feature_extractor.shape[0],-1)

    test_data = pd.DataFrame(features)
    return test_data

 
#accuracy = [0.95,0.94,0.975,0.945,0.965,0.96,0.95,0.96,0.955,0.955,0.965,0.995]
accuracy = [0.95,0.94,0.975,0.945,0.965,0.96]

def probability(predicted,accuracy):
    predicted = np.array(predicted)
    res = ''
    prob = 0
    result = np.bincount(predicted).argmax()
    for i in range(0,len(accuracy)):
        if (i < (len(accuracy)-3)):
            prob = prob + 1000*accuracy[i]*predicted[i]
        elif(i ==(len(accuracy) - 3)):
            prob = prob + 1116*accuracy[i]*predicted[i]
        else: 
            prob = prob + 3000*accuracy[i]*predicted[i]

    prob = (prob / 16116)*100
    if result == 1:
        res = 'positive'
    else:
        res = 'negative'
    return prob,res
def predict1(new_sample):
    res = ''
    predicted = []
    for model in models:
        out = model.predict(new_sample)
        predicted.append(out)
    predicted = [j for i in predicted for j in i]
    proba,result = probability(predicted,accuracy)
    proba = np.round(proba,4)
    if proba == 0: 
        proba = np.mean(accuracy)*100
    elif(proba <= 45):
        proba = 100- proba
    else:
        proba = proba
    return result,proba
    
app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')


@app.route("/prediction",methods = ["POST"])
def prediction():
    imagefile = request.files['img']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    new_sample = new_sample_feature(image_path)
    
    res,probab  = predict1(new_sample)




    
    return render_template('res.html', probab = probab,res = res )



if __name__ == '__main__':
    app.run(debug= True)
    