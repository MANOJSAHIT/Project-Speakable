from flask import Flask, render_template, Response
from keras.models import model_from_json
from string import ascii_uppercase
import numpy as np
import operator
import json
import cv2
import os
import sys

ASL_json_file = open("model/ASL-model-bw.json", "r")
ASL_model_json = ASL_json_file.read()
ASL_json_file.close()
ASL_loaded_model = model_from_json(ASL_model_json)
ASL_loaded_model.load_weights("model/ASL-model-bw.h5")

ASL_json_file_dru = open("model/ASL-model-bw_dru.json" , "r")
ASL_model_json_dru = ASL_json_file_dru.read()
ASL_json_file_dru.close()
ASL_loaded_model_dru = model_from_json(ASL_model_json_dru)
ASL_loaded_model_dru.load_weights("model/ASL-model-bw_dru.h5")

ASL_json_file_tkdi = open("model/ASL-model-bw_tkdi.json" , "r")
ASL_model_json_tkdi = ASL_json_file_tkdi.read()
ASL_json_file_tkdi.close()
ASL_loaded_model_tkdi = model_from_json(ASL_model_json_tkdi)
ASL_loaded_model_tkdi.load_weights("model/ASL-model-bw_tkdi.h5")

ASL_json_file_smn = open("model/ASL-model-bw_smn.json" , "r")
ASL_model_json_smn = ASL_json_file_smn.read()
ASL_json_file_smn.close()
ASL_loaded_model_smn = model_from_json(ASL_model_json_smn)
ASL_loaded_model_smn.load_weights("model/ASL-model-bw_smn.h5")
    
# ISL_json_file = open("model/ISL-model-bw.json", "r")
# ISL_model_json = ISL_json_file.read()
# ISL_json_file.close()
# ISL_loaded_model = model_from_json(ISL_model_json)
# ISL_loaded_model.load_weights("model/ISL-model-bw.h5")

def predictASL(res):
    global ASL_loaded_model
    global ASL_loaded_model_dru
    global ASL_loaded_model_tkdi
    global ASL_loaded_model_smn
    global word
    global sentence
    global ct
    global blank_flag
    global current_symbol
    
    res = cv2.resize(res, (128,128))
    result = ASL_loaded_model.predict(res.reshape(1, 128, 128, 1))
    result_dru = ASL_loaded_model_dru.predict(res.reshape(1 , 128 , 128 , 1))
    result_tkdi = ASL_loaded_model_tkdi.predict(res.reshape(1 , 128 , 128 , 1))
    result_smn = ASL_loaded_model_smn.predict(res.reshape(1 , 128 , 128 , 1))
    prediction={}
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    #LAYER 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]
    #LAYER 2
    if(current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U'):
        prediction = {}
        prediction['D'] = result_dru[0][0]
        prediction['R'] = result_dru[0][1]
        prediction['U'] = result_dru[0][2]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
    if(current_symbol == 'D' or current_symbol == 'I' or current_symbol == 'K' or current_symbol == 'T'):
        prediction = {}
        prediction['D'] = result_tkdi[0][0]
        prediction['I'] = result_tkdi[0][1]
        prediction['K'] = result_tkdi[0][2]
        prediction['T'] = result_tkdi[0][3]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
    if(current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'S'):
        prediction1 = {}
        prediction1['M'] = result_smn[0][0]
        prediction1['N'] = result_smn[0][1]
        prediction1['S'] = result_smn[0][2]
        prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        if(prediction1[0][0] == 'S'):
            current_symbol = prediction1[0][0]
        else:
            current_symbol = prediction[0][0]
    if(current_symbol == 'blank'):
        for i in ascii_uppercase:
            ct[i] = 0
    ct[current_symbol] += 1
    if(ct[current_symbol] > 60):
        for i in ascii_uppercase:
            if i == current_symbol:
                continue
            tmp = ct[current_symbol] - ct[i]
            if tmp < 0:
                tmp *= -1
            if tmp <= 20:
                ct['blank'] = 0
                for i in ascii_uppercase:
                    ct[i] = 0
                return
        ct['blank'] = 0
        for i in ascii_uppercase:
            ct[i] = 0
        if current_symbol == 'blank':
            if blank_flag == 0:
                blank_flag = 1
                if len(sentence) > 0:
                    sentence += " "
                sentence += word
                word = ""
        else:
            if(len(sentence) > 16):
                sentence = ""
            blank_flag = 0
            word += current_symbol

def predictISL(res):
    global ISL_loaded_model
    global word
    global sentence
    global ct
    global blank_flag
    global current_symbol
    
    res = cv2.resize(res, (128,128))
    result = ISL_loaded_model.predict(res.reshape(1, 128, 128, 1))
    prediction={}
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]
    if(current_symbol == 'blank'):
        for i in ascii_uppercase:
            ct[i] = 0
    ct[current_symbol] += 1
    if(ct[current_symbol] > 60):
        for i in ascii_uppercase:
            if i == current_symbol:
                continue
            tmp = ct[current_symbol] - ct[i]
            if tmp < 0:
                tmp *= -1
            if tmp <= 20:
                ct['blank'] = 0
                for i in ascii_uppercase:
                    ct[i] = 0
                return
        ct['blank'] = 0
        for i in ascii_uppercase:
            ct[i] = 0
        if current_symbol == 'blank':
            if blank_flag == 0:
                blank_flag = 1
                if len(sentence) > 0:
                    sentence += " "
                sentence += word
                word = ""
        else:
            if(len(sentence) > 16):
                sentence = ""
            blank_flag = 0
            word += current_symbol

ct = {}
ct['blank'] = 0
blank_flag = 0
for i in ascii_uppercase:
    ct[i] = 0
sentence = ""
word = ""
current_symbol = "Empty"
dataset = 1
stop = False

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/startRecognising')
def startRecognising():
    global stop
    global dataset
    vs = cv2.VideoCapture(0)
    while(vs.isOpened()):
        ok, frame = vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),0] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
            cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),1] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
            cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),2] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
            cv2.imwrite("static/frame.jpg", cv2image)
            if dataset==1:
                predictASL(cv2.flip(res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])],1))
            elif dataset==2:
                predictISL(cv2.flip(res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])],1))
            if stop:
                stop=False
                break
    vs.release()
    cv2.destroyAllWindows()
    return 'Success'
@app.route('/getData')
def getData():
    global current_symbol
    global word
    global sentence
    data = {}
    data['character'] = current_symbol
    data['word'] = word
    data['sentence'] = sentence
    return json.dumps(data)
@app.route('/changeToASL')
def changeToASL():
    global dataset
    dataset = 1
    return 'Success'
@app.route('/changeToISL')
def changeToISL():
    global dataset
    dataset = 2
    return 'Success'
@app.route('/stopRecognising')
def stopRecognising():
    global stop
    stop = True
    return 'Success'
if __name__ == '__main__':
    app.run()
