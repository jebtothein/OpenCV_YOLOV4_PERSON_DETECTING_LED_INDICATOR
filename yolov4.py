import cv2 as cv
import time
import serial #for Serial communication
arduino = serial.Serial('COM3',9600) #Create Serial port object called arduinoSerialData

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)] # to define the colours for each individual class

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)(basically to list and shows all of the classes)
#-----------------------------------------------------------------------------------------(this part is basically just for classes and to differenciate the colours)------------------------------------------------------------------------------------------------------------------------------

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg') 
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16) #to set the target

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
#-----------------------------------------------------------------------------------------(this part is basically just to set up the paramaters for the detector)------------------------------------------------------------------------------------------------------------------------------

net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg') 
cap = cv.VideoCapture(0) #taking the video or the webcom footage (the basic)
starting_time = time.time()
frame_counter = 0
while True: # The basic
    ret, frame = cap.read() # The basic
    frame_counter += 1
    if ret == False: # The basic
        break # The basic
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold) #parameter required for object detector
    
    try:
        for (classid, score, box) in zip(classes, scores, boxes):#coordinate and score aand boxes for each object that was detected
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_name[classid], score)
            cv.rectangle(frame, box, color, 1) #the rectangle surrounding the detected object
            cv.putText(frame, label, (box[0], box[1]-10),# labeling each boxes
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
            print ('object detected',box)
            arduino.write(b'H') 
    except:
        pass
        arduino.write(b'L')
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
#---------------------------------------------------------------------------------------------(this part is for the object detector)--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cv.imshow('frame', frame) # The basic
    key = cv.waitKey(1) 
    if key == ord('q'): # The basic (to show the display)

        break # The basic
       
cap.release() # The basic
cv.destroyAllWindows()
