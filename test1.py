from ultralytics import YOLO
import cv2
import time
import keyboard

# Load a model
# model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('./model/yolov8n-pose.pt')  # load a custom model
data = cv2.imread('./image.jpg')
source = './image.jpg'
cnt = 0
start_time = time.time()
for i in range(1000):
    # start_time = time.time()
    # image = model(data)[0].plot()
    result = model(data, half = True)
    cnt += 1
    # print(result)
    # end_time = time.time()
    if keyboard.is_pressed('q'):
        break
    # print(str(1 // (end_time - start_time)))
print(1 // ((time.time() - start_time) / cnt))