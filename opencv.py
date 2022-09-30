import cv2
import glob
import os

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

os.chdir('./downloads')
name_list = os.listdir()
os.chdir('../')

for name in name_list:
    path = f'/Users/mok/Section4_Project/downloads/{name}/*'
    img_number = 1
    img_list = glob.glob(path)

    for file in img_list:
        print(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        try:
            for (x,y,w,h) in faces:
                roi_color = img[y:y+h, x:x+w]
            resized = cv2.resize(roi_color, (224,224))
            cv2.imwrite(f"extracted_faces2/"+f"{name}_" + str(img_number)+".jpg", resized)
        except:
            print("No faces detected")
        img_number += 1