import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import dlib
from pylepton import Lepton
import pandas as pd





#Initialization of Camera/Windows
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.rotation = 180
rawCapture = PiRGBArray(camera, size=(640, 480))
lepton_buf = np.zeros((60,80,1), dtype=np.uint16)
cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Thermal', 400,300)
cv2.namedWindow("PyCam", cv2.WINDOW_NORMAL)
cv2.resizeWindow('PyCam', 400,300)

# define transform
# h = np.float32([[ 2.24346513e+00,  6.48002063e-01, -1.69435974e+02],
#  [ 7.40627465e-02,  2.71901217e+00, -3.16027302e+02],
#  [ 1.35883889e-04,  2.71327283e-03,  1.00000000e+00]])

h = np.float32(np.load('trans_param.npy')) # get transform parameters from file

# create data frame for data
DF = pd.DataFrame(data=None, columns=['Face','LM1','LM2','LM3','LM4','LM5'])
    
    
#Allow Camera to warm up
time.sleep(0.1)

## dlib face detector setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

## define normalization alpha and beta
alpha = -70000
beta = 70000

## define translation parameters
x_pos = 0
y_pos = 0

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    with Lepton() as l:
        a,_ = l.capture()
        cv2.normalize(a, a, alpha, beta, cv2.NORM_MINMAX) # extend contrast
        a = np.rot90(a, 2)
        a = (a/256).astype('uint8')
        
        def on_click(event, x, y, p1, p2):
           if event == cv2.EVENT_LBUTTONDOWN:
            print(a[x,y])

        # update translation matrix
        translation_matrix = np.float32([[1,0,x_pos],[0,1,y_pos]])

        image = frame.array
        image = cv2.resize(image, (400,300))
        a = cv2.resize(a, (400, 300))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        # define array for data storage
        face_num = 0
        # translate non thermal iamge
        image = cv2.warpAffine(image, translation_matrix, (400,300))
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(image, face)
            
            thermal_data = []
            for n in range(0, 5):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
                thermal_pixel = a[x,y]
                if n < 2:
                    cv2.putText(image, str(thermal_pixel), (int(x*1.1),int(y*1.1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                else:
                    cv2.putText(image, str(thermal_pixel), (int(x*.9),int(y*1.1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                thermal_data.append(thermal_pixel)
                
            DF = DF.append({'Face': face_num,
                            'LM1': thermal_data[0],
                            'LM2': thermal_data[1],
                            'LM3': thermal_data[2],
                            'LM4': thermal_data[3],
                            'LM5': thermal_data[4]},
                           ignore_index=True)
            face_num += 1
            
            
        cv2.setMouseCallback('Thermal', on_click)
        # show the frame
        cv2.imshow("PyCam", image)
        cv2.imshow('Thermal', a)
        color_map = cv2.applyColorMap(a, cv2.COLORMAP_JET)
        cv2.imshow("color",color_map)
        # show warped image
        warp_src = cv2.warpPerspective(image, h, (400,300)) # apply perspective warp
#         cv2.imshow("warp",warp_src)
        # show overlay
        a_3 = cv2.merge((a,a,a))
        blnd = cv2.addWeighted(a_3,0.7,warp_src,0.3,0)
        cv2.imshow("blnd",blnd)
        key = cv2.waitKey(1) & 0xFF
        #o.update(np.getbuffer(a))
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord('t'):
            alpha += 5000
            print("Alpha is %d" % (alpha,))
        if key == ord('g'):
            alpha -= 5000
            print("Alpha is %d" % (alpha,))
        if key == ord('y'):
            beta += 5000
            print("Beta is %d" % (beta,))
        if key == ord('h'):
            beta -= 5000
            print("Beta is %d" % (beta,))
        if key == ord('j'):
            x_pos += -1
            print("x_pos %d" % (x_pos,))
        if key == ord('l'):
            x_pos += 1
            print("x_pos %d" % (x_pos,))
        if key == ord('i'):
            y_pos += 1
            print("y_pos %d" % (y_pos,))
        if key == ord('k'):
            y_pos -= 1
            print("y_pos %d" % (y_pos,))
            
        if key == ord("q"):
            DF.to_csv('./data/face_data_excersize.csv')
            print(a)
            break
################
