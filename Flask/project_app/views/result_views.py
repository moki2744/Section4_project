from flask import Blueprint, render_template, request, url_for
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import numpy as np

# 학습된 model 가져오기
new_model = tf.keras.models.load_model('project_app/views/vgg_face_trained_model.h5')

# Blueprint
result_bp = Blueprint('predict', __name__, url_prefix='/result')

# label
label1 = {0: '안효섭', 1: '배수지', 2: '채시라', 3: '차은우', 4: '최지우', 5: '고아라', 6: '고두심', 7: '고현정', 8: '공유', 9: '고소영', 10: '하지원', 11: '한채영', 12: '한가인', 13: '한지민', 14: '한예슬', 15: '황신혜', 16: '현빈', 17: '장근석', 18: '전지현', 19: '지창욱', 20: '조인성', 21: '정해인', 22: '강하늘', 23: '김범', 24: '김하늘', 25: '김희애', 26: '김희선', 27: '김혜수', 28: '김현주', 29: '김현중', 30: '김남주', 31: '김수현', 32: '김태희', 33: '김우빈', 34: '이동욱', 35: '이종석', 36: '이준기', 37: '이민호', 38: '이민정', 39: '이나영', 40: '이승기', 41: '이연희', 42: '이영애', 43: '문채원', 44: '문근영', 45: '남주혁', 46: '박보검', 47: '박해진', 48: '박형식', 49: '박서준', 50: '박신혜', 51: '서강준', 52: '서인국', 53: '신민아', 54: '신세경', 55: '소지섭', 56: '송강', 57: '송혜교', 58: '송중기', 59: '송승헌', 60: '손예진', 61: '수애', 62: '우도환', 63: '유승호', 64: '육성재'}

# label
label2 = {0: 'anhyoseob', 1: 'baesuji', 2: 'chaesira', 3: 'chaeunwoo', 4: 'choijiwoo', 5: 'goara', 6: 'godusim', 7: 'gohyunjung', 8: 'gongyou', 9: 'gosoyoung', 10: 'hajiwon', 11: 'hanchaeyoung', 12: 'hangain', 13: 'hanjimin', 14: 'hanyeseul', 15: 'hwangsinhye', 16: 'hyunbin', 17: 'janggeunsuck', 18: 'jeonjihyun', 19: 'jichangwook', 20: 'joinsung', 21: 'junghaein', 22: 'kanghanul', 23: 'kimbum', 24: 'kimhaneul', 25: 'kimheeae', 26: 'kimheesun', 27: 'kimhyesu', 28: 'kimhyunjoo', 29: 'kimhyunjoong', 30: 'kimnamju', 31: 'kimsuhyun', 32: 'kimtaehee', 33: 'kimwoobin', 34: 'leedongwook', 35: 'leejongseok', 36: 'leejungi', 37: 'leeminho', 38: 'leeminjung', 39: 'leenayoung', 40: 'leeseunggi', 41: 'leeyeonhee', 42: 'leeyoungae', 43: 'mooncheawon', 44: 'moongeunyoung', 45: 'namjuhyuk', 46: 'parkbogum', 47: 'parkhaejin', 48: 'parkhyungsik', 49: 'parkseojun', 50: 'parksinhye', 51: 'seogangjun', 52: 'seoingook', 53: 'shinmina', 54: 'shinsekyung', 55: 'sojiseob', 56: 'songgang', 57: 'songhyekyo', 58: 'songjungki', 59: 'songseunghun', 60: 'sonyejin', 61: 'sooae', 62: 'woodohwan', 63: 'youseongho', 64: 'yuksungjae'}

# /result 주소화면    
@result_bp.route('/',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        f = request.files['user_file']
        d = os.path.join('project_app/static', secure_filename(f.filename))
        f.save(d)
        pred = predict(d)
        pred_name1 = label1.get(pred)
        pred_name2 = label2.get(pred)
        name = pred_name2
        root = f'project_app/static/downloads/{name}'
        file = os.listdir(root)
        file_path = f'/downloads/{name}/{file}'
        return render_template("result.html", pred_name1 = pred_name1, file_path = file_path)
    else:
        return render_template("index.html")

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def predict(directory):
    pixels = extract_face(directory)
    pixels = pixels.astype('float32')
    samples = np.expand_dims(pixels, axis=0) / 255.
    result = new_model.predict(samples)
    result_label = np.argmax(result)
    return result_label