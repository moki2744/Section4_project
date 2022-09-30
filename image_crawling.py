# -*- coding: utf-8 -*-

# google_images_download 라이브러리를 이용한 이미지 크롤링 영역
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords":"하희라, 한채영, 이나영, 채시라, 김혜자, 김현주, 김남주, 문근영, 신세경, 최지우, 고두심, 신민아, 이연희, 수애, 이민정, 김하늘, 한예슬, 하지원, 한지민, 고아라, 박신혜, 문채원, 배수지, 황신혜, 심은하, 전지현, 고소영, 한가인, 김희애, 김혜수, 고현정, 손예진, 송혜교, 이영애, 김희선, 김태희, 조인성, 서인국, 장근석, 송강, 우도환, 정해인, 강하늘, 소지섭, 유승호, 남주혁, 송중기, 박해진, 공유, 김우빈, 박보검, 박형식, 안효섭, 송승헌, 김범, 육성재, 이동욱, 현빈, 차은우, 서강준, 박서준, 이종석, 이승기, 이준기, 지창욱, 이민호, 김현중, 김수현", "limit":50,"print_urls":True, "format" : "jpg"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images