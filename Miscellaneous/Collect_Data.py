import numpy as np
import string
import cv2
import os
from scipy import ndimage
from scipy.ndimage import convolve
from scipy import misc

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")
if not os.path.exists("data/train/0"):
    os.makedirs("data/train/0")
if not os.path.exists("data/test/0"):
    os.makedirs("data/test/0")
for i in string.ascii_uppercase:
    if not os.path.exists("data/train/" + i):
        os.makedirs("data/train/"+i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/"+i)
if not os.path.exists("dataAT"):
    os.makedirs("dataAT")
if not os.path.exists("dataAT/train"):
    os.makedirs("dataAT/train")
if not os.path.exists("dataAT/test"):
    os.makedirs("dataAT/test")
if not os.path.exists("dataAT/train/0"):
    os.makedirs("dataAT/train/0")
if not os.path.exists("dataAT/test/0"):
    os.makedirs("dataAT/test/0")
for i in string.ascii_uppercase:
    if not os.path.exists("dataAT/train/" + i):
        os.makedirs("dataAT/train/"+i)
    if not os.path.exists("dataAT/test/" + i):
        os.makedirs("dataAT/test/"+i)
if not os.path.exists("dataCE"):
    os.makedirs("dataCE")
if not os.path.exists("dataCE/train"):
    os.makedirs("dataCE/train")
if not os.path.exists("dataCE/test"):
    os.makedirs("dataCE/test")
if not os.path.exists("dataCE/train/0"):
    os.makedirs("dataCE/trainCE/0")
if not os.path.exists("dataCE/test/0"):
    os.makedirs("dataCE/test/0")
for i in string.ascii_uppercase:
    if not os.path.exists("dataCE/train/" + i):
        os.makedirs("dataCE/train/"+i)
    if not os.path.exists("dataCE/test/" + i):
        os.makedirs("dataCE/test/"+i)

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

mode = 'train'
directory = 'data/'+mode+'/'
directory1 = 'dataAT/'+mode+'/'
directory2 = 'dataCE/'+mode+'/'
minValue = 70
cap = cv2.VideoCapture(0)
interrupt = -1
while (cap.isOpened()):
    ok, frame = cap.read()
    if ok:
        frame = cv2.flip(frame, 1)
        count = {
            'zero': len(os.listdir(directory+"0")),
            'a': len(os.listdir(directory+"A")),
            'b': len(os.listdir(directory+"B")),
            'c': len(os.listdir(directory+"C")),
            'd': len(os.listdir(directory+"D")),
            'e': len(os.listdir(directory+"E")),
            'f': len(os.listdir(directory+"F")),
            'g': len(os.listdir(directory+"G")),
            'h': len(os.listdir(directory+"H")),
            'i': len(os.listdir(directory+"I")),
            'j': len(os.listdir(directory+"J")),
            'k': len(os.listdir(directory+"K")),
            'l': len(os.listdir(directory+"L")),
            'm': len(os.listdir(directory+"M")),
            'n': len(os.listdir(directory+"N")),
            'o': len(os.listdir(directory+"O")),
            'p': len(os.listdir(directory+"P")),
            'q': len(os.listdir(directory+"Q")),
            'r': len(os.listdir(directory+"R")),
            's': len(os.listdir(directory+"S")),
            't': len(os.listdir(directory+"T")),
            'u': len(os.listdir(directory+"U")),
            'v': len(os.listdir(directory+"V")),
            'w': len(os.listdir(directory+"W")),
            'x': len(os.listdir(directory+"X")),
            'y': len(os.listdir(directory+"Y")),
            'z': len(os.listdir(directory+"Z"))
        }
        cv2.putText(frame, "0 : "+str(count['zero']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "c : "+str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "d : "+str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "e : "+str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "f : "+str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "g : "+str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "h : "+str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "i : "+str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "j : "+str(count['j']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "k : "+str(count['k']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "l : "+str(count['l']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "m : "+str(count['m']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "n : "+str(count['n']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "o : "+str(count['o']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "p : "+str(count['p']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "q : "+str(count['q']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "r : "+str(count['r']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "s : "+str(count['s']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "t : "+str(count['t']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "u : "+str(count['u']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "v : "+str(count['v']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "w : "+str(count['w']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "x : "+str(count['x']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "y : "+str(count['y']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "z : "+str(count['z']), (10, 350), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.rectangle(frame, (int(0.6*frame.shape[1]), 0), (frame.shape[1], int(0.5*frame.shape[0])), (255,0,0) ,1)
        cv2.imshow("Frame", frame)
        roi = frame[0:int(0.5*frame.shape[0]), int(0.6*frame.shape[1]):frame.shape[1]]
        cv2.imshow("Image to be saved", roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        test_image = cv2.resize(test_image, (300,300))
        cv2.imshow("Image to be saved AT", test_image)
        gray2 = rgb2gray(roi)
        image2 = detect(gray2)
        test_image2 = image2.astype(np.uint8)
        test_image2 = cv2.resize(test_image2, (300,300))
        cv2.imshow("Image to be saved CE", test_image2)
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
            cv2.imwrite(directory1+'0/'+str(count['zero'])+'.jpg', test_image)
            cv2.imwrite(directory2+'0/'+str(count['zero'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('a'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg', roi)
            cv2.imwrite(directory1+'A/'+str(count['a'])+'.jpg', test_image)
            cv2.imwrite(directory2+'A/'+str(count['a'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('b'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg', roi)
            cv2.imwrite(directory1+'B/'+str(count['b'])+'.jpg', test_image)
            cv2.imwrite(directory2+'B/'+str(count['b'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('c'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg', roi)
            cv2.imwrite(directory1+'C/'+str(count['c'])+'.jpg', test_image)
            cv2.imwrite(directory2+'C/'+str(count['c'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('d'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg', roi)
            cv2.imwrite(directory1+'D/'+str(count['d'])+'.jpg', test_image)
            cv2.imwrite(directory2+'D/'+str(count['d'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('e'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg', roi)
            cv2.imwrite(directory1+'E/'+str(count['e'])+'.jpg', test_image)
            cv2.imwrite(directory2+'E/'+str(count['e'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('f'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg', roi)
            cv2.imwrite(directory1+'F/'+str(count['f'])+'.jpg', test_image)
            cv2.imwrite(directory2+'F/'+str(count['f'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('g'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg', roi)
            cv2.imwrite(directory1+'G/'+str(count['g'])+'.jpg', test_image)
            cv2.imwrite(directory2+'G/'+str(count['g'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('h'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg', roi)
            cv2.imwrite(directory1+'H/'+str(count['h'])+'.jpg', test_image)
            cv2.imwrite(directory2+'H/'+str(count['h'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('i'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg', roi)
            cv2.imwrite(directory1+'I/'+str(count['i'])+'.jpg', test_image)
            cv2.imwrite(directory2+'I/'+str(count['i'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('j'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg', roi)
            cv2.imwrite(directory1+'J/'+str(count['j'])+'.jpg', test_image)
            cv2.imwrite(directory2+'J/'+str(count['j'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('k'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg', roi)
            cv2.imwrite(directory1+'K/'+str(count['k'])+'.jpg', test_image)
            cv2.imwrite(directory2+'K/'+str(count['k'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('l'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg', roi)
            cv2.imwrite(directory1+'L/'+str(count['l'])+'.jpg', test_image)
            cv2.imwrite(directory2+'L/'+str(count['l'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('m'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg', roi)
            cv2.imwrite(directory1+'M/'+str(count['m'])+'.jpg', test_image)
            cv2.imwrite(directory2+'M/'+str(count['m'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('n'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg', roi)
            cv2.imwrite(directory1+'N/'+str(count['n'])+'.jpg', test_image)
            cv2.imwrite(directory2+'N/'+str(count['n'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('o'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg', roi)
            cv2.imwrite(directory1+'O/'+str(count['o'])+'.jpg', test_image)
            cv2.imwrite(directory2+'O/'+str(count['o'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('p'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg', roi)
            cv2.imwrite(directory1+'P/'+str(count['p'])+'.jpg', test_image)
            cv2.imwrite(directory2+'P/'+str(count['p'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('q'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg', roi)
            cv2.imwrite(directory1+'Q/'+str(count['q'])+'.jpg', test_image)
            cv2.imwrite(directory2+'Q/'+str(count['q'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('r'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg', roi)
            cv2.imwrite(directory1+'R/'+str(count['r'])+'.jpg', test_image)
            cv2.imwrite(directory2+'R/'+str(count['r'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('s'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg', roi)
            cv2.imwrite(directory1+'S/'+str(count['s'])+'.jpg', test_image)
            cv2.imwrite(directory2+'S/'+str(count['s'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('t'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg', roi)
            cv2.imwrite(directory1+'T/'+str(count['t'])+'.jpg', test_image)
            cv2.imwrite(directory2+'T/'+str(count['t'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('u'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg', roi)
            cv2.imwrite(directory1+'U/'+str(count['u'])+'.jpg', test_image)
            cv2.imwrite(directory2+'U/'+str(count['u'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('v'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg', roi)
            cv2.imwrite(directory1+'V/'+str(count['v'])+'.jpg', test_image)
            cv2.imwrite(directory2+'V/'+str(count['v'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('w'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg', roi)
            cv2.imwrite(directory1+'W/'+str(count['w'])+'.jpg', test_image)
            cv2.imwrite(directory2+'W/'+str(count['w'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('x'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg', roi)
            cv2.imwrite(directory1+'X/'+str(count['x'])+'.jpg', test_image)
            cv2.imwrite(directory2+'X/'+str(count['x'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('y'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg', roi)
            cv2.imwrite(directory1+'Y/'+str(count['y'])+'.jpg', test_image)
            cv2.imwrite(directory2+'Y/'+str(count['y'])+'.jpg', test_image2)
        if interrupt & 0xFF == ord('z'):
            roi = cv2.resize(roi, (300,300))
            cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg', roi)
            cv2.imwrite(directory1+'Z/'+str(count['z'])+'.jpg', test_image)
            cv2.imwrite(directory2+'Z/'+str(count['z'])+'.jpg', test_image2)
cap.release()
cv2.destroyAllWindows()
