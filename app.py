from flask import Flask, render_template, request
from keras.models import model_from_json
from string import ascii_uppercase
import numpy as np
import operator
import json
import cv2
import os
import sys
import base64
import io
from imageio.v2 import imread
from scipy import ndimage
from scipy.ndimage import convolve
from scipy import misc

sigma=1.4
kernel_size=5
lowthreshold=0.09
highthreshold=0.17
strong_pixel=255
weak_pixel=100
img_smoothed = None
gradientMat = None
thetaMat = None
nonMaxImg = None
thresholdImg = None

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                pass
    return Z

def threshold(img):
    global highthreshold
    global lowthreshold
    global strong_pixel
    global weak_pixel
    highThresholdP = img.max() * highthreshold;
    lowThresholdP = highThresholdP * lowthreshold;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)
    strong_i, strong_j = np.where(img >= highThresholdP)
    zeros_i, zeros_j = np.where(img < lowThresholdP)
    weak_i, weak_j = np.where((img <= highThresholdP) & (img >= lowThresholdP))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res)

def hysteresis(img):
    global strong_pixel
    global weak_pixel
    M, N = img.shape
    weak = weak_pixel
    strong = strong_pixel
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def detect(img):
    img_smoothed = convolve(img, gaussian_kernel(kernel_size, sigma))
    gradientMat, thetaMat = sobel_filters(img_smoothed)
    nonMaxImg = non_max_suppression(gradientMat, thetaMat)
    thresholdImg = threshold(nonMaxImg)
    img_final = hysteresis(thresholdImg)
    return img_final

ASL_json_file = open("model/ASL-model-dataAT.json", "r")
ASL_model_json = ASL_json_file.read()
ASL_json_file.close()
ASL_loaded_model = model_from_json(ASL_model_json)
ASL_loaded_model.load_weights("model/ASL-model-dataAT.h5")

ASL_json_file_dru = open("model/ASL-model-dataAT_dru.json" , "r")
ASL_model_json_dru = ASL_json_file_dru.read()
ASL_json_file_dru.close()
ASL_loaded_model_dru = model_from_json(ASL_model_json_dru)
ASL_loaded_model_dru.load_weights("model/ASL-model-dataAT_dru.h5")

ASL_json_file_tkdi = open("model/ASL-model-dataAT_tkdi.json" , "r")
ASL_model_json_tkdi = ASL_json_file_tkdi.read()
ASL_json_file_tkdi.close()
ASL_loaded_model_tkdi = model_from_json(ASL_model_json_tkdi)
ASL_loaded_model_tkdi.load_weights("model/ASL-model-dataAT_tkdi.h5")

ASL_json_file_smn = open("model/ASL-model-dataAT_smn.json" , "r")
ASL_model_json_smn = ASL_json_file_smn.read()
ASL_json_file_smn.close()
ASL_loaded_model_smn = model_from_json(ASL_model_json_smn)
ASL_loaded_model_smn.load_weights("model/ASL-model-dataAT_smn.h5")
    
ISL_json_file = open("model/ISL-model-data.json", "r")
ISL_model_json = ISL_json_file.read()
ISL_json_file.close()
ISL_loaded_model = model_from_json(ISL_model_json)
ISL_loaded_model.load_weights("model/ISL-model-data.h5")

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
    if(ct[current_symbol] > 30):
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

def predictISL(rs): #Under Construction...to be integrated by another team
    global ISL_loaded_model
    global word
    global sentence
    global ct
    global blank_flag
    global current_symbol
    
    res = rs.astype(np.uint8)
    res = cv2.resize(res, (261,310))
    result = ISL_loaded_model.predict(res.reshape(1, 261, 310, 1))
    prediction={}
    prediction['0'] = result[0][0]
    prediction['1'] = result[0][1]
    prediction['2'] = result[0][2]
    prediction['3'] = result[0][3]
    prediction['4'] = result[0][4]
    prediction['5'] = result[0][5]
    prediction['6'] = result[0][6]
    prediction['7'] = result[0][7]
    prediction['8'] = result[0][8]
    prediction['9'] = result[0][9]
    inde = 10
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]
    word = "Cannot generate words in ISL"
    sentence = "Cannot generate sentence in ISL"

ct = {}
ct['blank'] = 0
blank_flag = 0
for i in ascii_uppercase:
    ct[i] = 0
sentence = ""
word = ""
current_symbol = "Empty"

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/recognise', methods=['POST'])
def recognise():
    global current_symbol
    global word
    global sentence
    imageData = request.get_json().get('imageData')
    dataset = request.get_json().get('dataset')
    imageData = bytes(imageData[23:], 'utf-8')
    img = imread(io.BytesIO(base64.b64decode(imageData)))
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2image = cv2.flip(frame, 1)
    if dataset==1:
        gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),0] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
        cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),1] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
        cv2image[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1]),2] = res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])]
        predictASL(cv2.flip(res[0:int(0.5*res.shape[0]),int(0.6*res.shape[1]):int(res.shape[1])],1))
    elif dataset==2:
        gray2 = rgb2gray(cv2image)
        res2 = detect(gray2)
        cv2image[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1]),0] = res2[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1])]
        cv2image[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1]),1] = res2[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1])]
        cv2image[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1]),2] = res2[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1])]
        predictISL(cv2.flip(res2[0:int(0.5*res2.shape[0]),int(0.6*res2.shape[1]):int(res2.shape[1])],1))
    retval, buffer = cv2.imencode('.png', cv2image)
    png_as_text = base64.b64encode(buffer)
    data = {}
    data['character'] = current_symbol
    data['word'] = word
    data['sentence'] = sentence
    data["adaptiveImg"] = "data:image/png;base64, "+(png_as_text.decode('utf-8'))
    return json.dumps(data)
if __name__ == '__main__':
    app.run('localhost', 5000)
