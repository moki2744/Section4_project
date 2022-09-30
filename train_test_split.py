import shutil
import os


def make_testset(src_dir, des_dir):
    for f in os.listdir(src_dir): #--> ['이름','이름','이름']
        test_path=os.path.join(des_dir,f) # test_path : test directory + '이름'의 새로운 directory
        train_path=os.path.join(src_dir,f) # train_path : train directory + '이름'의 directory
        if not os.path.exists(test_path): # 해당 경로가 존재하지 않는다면
            os.makedirs(test_path)        # test 폴더에 각각 연예인 이름의 폴더를 만들어준다.
            for t in os.listdir(train_path)[-4:-1]:
                shutil.move(os.path.join(train_path, t), os.path.join(test_path, t))
    return

src_dir= 'C:/Users/mok/Section4_Project/face_detect_train'
des_dir = 'C:/Users/mok/Section4_Project/face_detect_test'

make_testset(src_dir, des_dir)



