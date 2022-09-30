from googleapiclient import discovery
from oauth2client.client  import GoogleCredentials
import sys
import io
import base64
from PIL import Image
from PIL import ImageDraw
from genericpath import isfile
import os
import hashlib
from oauth2client.service_account import ServiceAccountCredentials


NUM_THREADS = 10
MAX_FACE = 2
MAX_LABEL = 50
IMAGE_SIZE = 224,224
MAX_ROLL = 90
MAX_TILT = 90
MAX_PAN = 90

# index to transfrom image string label to number
global_label_index = 0 
global_label_number = [0 for x in range(1000)]
global_image_hash = []

class FaceDetector():
    # googleapi 초기화 영역
    def __init__(self):

        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
                        'C:/Users/mok/Section4_Project/golden-plateau-363802-73552b658459.json', scopes=scopes)
        self.service = discovery.build('vision', 'v1', credentials=credentials)
    
    # 이미지 파일로부터 얼굴 영역을 찾아 좌표를 반환한다.
    def detect_face(self,image_file):
        try:
            with io.open(image_file,'rb') as fd:
                image = fd.read()
                batch_request = [{
                        'image':{
                            'content':base64.b64encode(image).decode('utf-8')
                            },
                        'features':[
                            {
                            'type':'FACE_DETECTION',
                            'maxResults':MAX_FACE,
                            },
                            {
                            'type':'LABEL_DETECTION',
                            'maxResults':MAX_LABEL,
                            }
                                    ]
                        }]
                fd.close()
        
            request = self.service.images().annotate(body={
                            'requests':batch_request, })
            response = request.execute()
            if 'faceAnnotations' not in response['responses'][0]:
                 print('[Error] %s: Cannot find face ' % image_file)
                 return None
                
            face = response['responses'][0]['faceAnnotations']
            label = response['responses'][0]['labelAnnotations']
            
            if len(face) > 1 :
                print('[Error] %s: It has more than 2 faces in a file' % image_file)
                return None
            
            roll_angle = face[0]['rollAngle']
            pan_angle = face[0]['panAngle']
            tilt_angle = face[0]['tiltAngle']
            angle = [roll_angle,pan_angle,tilt_angle]
            
            # check angle
            # if face skew angle is greater than > 20, it will skip the data
            if abs(roll_angle) > MAX_ROLL or abs(pan_angle) > MAX_PAN or abs(tilt_angle) > MAX_TILT:
                print('[Error] %s: face skew angle is big' % image_file)
                return None
            
            # check sunglasses
            for l in label:
                if 'sunglasses' in l['description']:
                  print('[Error] %s: sunglass is detected' % image_file)  
                  return None
            
            box = face[0]['fdBoundingPoly']['vertices']
            left = box[0]['x']
            top = box[1]['y']
                
            right = box[2]['x']
            bottom = box[2]['y']
                
            rect = [left,top,right,bottom]
                
            print("[Info] %s: Find face from in position %s and skew angle %s" % (image_file,rect,angle))
            return rect
        except Exception as e:
            print('[Error] %s: cannot process file : %s' %(image_file,str(e)) )
    
    # 이미지 파일과 좌표를 받아서 -> 얼굴좌표만큼 잘라낸 뒤 -> 저장한다.
    def crop_face(self,image_file,rect,outputfile):
        
        global global_image_hash
        try:
            fd = io.open(image_file,'rb')
            image = Image.open(fd)  

            # extract hash from image to check duplicated image
            m = hashlib.md5()
            with io.BytesIO() as memf:
                image.save(memf, 'PNG')
                data = memf.getvalue()
                m.update(data)
            image_hash = m.hexdigest()
            
            if image_hash in global_image_hash:
                print('[Error] %s: Duplicated image' %(image_file) )
                return None
            global_image_hash.append(image_hash)

            crop = image.crop(rect)
            im = crop.resize(IMAGE_SIZE,Image.ANTIALIAS)
            
            
            im.save(outputfile,"JPEG")
            fd.close()
            print('[Info]  %s: Crop face %s and write it to file : %s' %( image_file,rect,outputfile) )
            return True
        except Exception as e:
            print('[Error] %s: Crop image writing error : %s' %(image_file,str(e)) )
    
    # 주어진 디렉토리에서 경로 + 파일명을 받아온다.
    def getfiles(self,src_dir):
        files = []
        for f in os.listdir(src_dir):                   #os.listdir(src_dir) : src_dir 경로에 있는 모든 폴더와 파일을 리스트 형태로 가져온다.
            if isfile(os.path.join(src_dir,f)):         #만약 파일이라면, files 리스트에 경로+파일명을 추가한다.
                if not f.startswith('.'):
                 files.append(os.path.join(src_dir,f))
        return files
    
    
    # 각각의 directory(연예인별)에서 사진을 가져와서 -> 얼굴 영역 작업을 마친 뒤 -> output directory에 저장시킨다
    def crop_faces_dir(self,src_dir,des_dir,maxnum):
        
        path,label = os.path.split(src_dir) #입력 이미지 경로를 head와 tail로 구분. 마지막 부분이 tail이며, 그외 모든부분을 head
        
        files = self.getfiles(src_dir) #입력 이미지의 경로+file명을 리스트 형태로 가져온다.
        global global_label_index
        cnt = 0 
        num = 0 # number of training data
        for f in files:
            rect = self.detect_face(f) #이미지 파일의 얼굴영역의 좌표를 받아온다.

            des_file_name = os.path.basename(f)             # 파일이름을 가져온다.
            des_file_name = des_file_name.replace(',','_')  # 파일이름 내 ','를 '_'로 치환한다.
            
            if rect != None:
                des_file = os.path.join(des_dir,des_file_name) 
                # if we already have duplicated image, crop_face will return None
                if self.crop_face(f, rect, des_file ) != None:
                    num = num + 1
                    global_label_number[global_label_index] = num
                    cnt = cnt+1
                    
                if (num>=maxnum):
                    break

                    
                if(cnt>100): 
                    cnt = 0
        #increase index for image label
        global_label_index = global_label_index + 1 
        print('## label %s has %s of training data' %(global_label_index,num))


    #downloads 폴더 내 각각의 연예인 폴더 directory를 리스트 형태로 반환.
    def getdirs(self,dir):
        dirs = []
        for f in os.listdir(dir):
            f=os.path.join(dir,f)
            if os.path.isdir(f):
                if not f.startswith('.'):
                    dirs.append(f)

        return dirs


    #output폴더도 input폴더구조로 만들어준다.
    def get_output_dirs(self,src_dir, des_dir):
        dirs = []
        for f in os.listdir(src_dir): #os.listdir() : 지정한 디렉토리 내 모든 파일과 디렉토리의 리스트를 리턴한다.
            f=os.path.join(des_dir,f) #os.path.join(dir, str) : dir + str을 진행하여 새로운 1개의 directory를 생성한다.
            if not os.path.exists(f): #os.path.exists(dir) : dir 이라는 directory 또는 file이 존재하는지 확인하여 bool 형태로 반환한다.
                os.makedirs(f)        #os.makedirs(dir) : dir 경로를 새로 만들어준다.
            if os.path.isdir(f):      #os.path.isdir(dir) : dir이라는 directory가 존재하는지 확인하여 bool 형태로 반환한다.
                if not f.startswith('.'):
                    dirs.append(f)
        return dirs
    

    # 시작 영역
    def crop_faces_rootdir(self,src_dir,des_dir,maxnum):

        src_dirs = self.getdirs(src_dir) # downloads 폴더 내 각각의 연예인 폴더 directory를 리스트 형태로 반환.
        des_dirs = self.get_output_dirs(src_dir, des_dir) # des_dir 내 폴더 생성(입력 데이터 directory와 동일하게) 후 directory를 리스트 형태로 반환.

        for s, d in zip(src_dirs, des_dirs):
            self.crop_faces_dir(s,d,maxnum)

        global global_label_number
        print("number of datas per label ", global_label_number)

#usage  

srcdir= 'C:/Users/mok/Section4_Project/downloads' # arg[1]
desdir = 'C:/Users/mok/Section4_Project/face_detect3' # arg[2]
maxnum = 500 # arg[3]

detector = FaceDetector()
detector.crop_faces_rootdir(srcdir, desdir, maxnum)