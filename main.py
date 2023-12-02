from multiprocessing import Process
from multiprocessing.sharedctypes import RawArray
from numpy import frombuffer
import numpy
import dxcam
import cv2
import dxcam
import time
import numpy as np
import cv2
from ultralytics import YOLO

def task1(array):
    target_fps = 170
    camera = dxcam.create(device_idx=0, output_idx=0, output_color="RGB")
    camera.start(target_fps=target_fps, video_mode=True)
    data = frombuffer(array, dtype=numpy.uint8, count=len(array))
    data = data.reshape((1440,2560,3))
    cnt = 0
    total = 0
    while True:
        start_time = time.time()
        data[:] = camera.get_latest_frame()
        end_time = time.time()
        cnt += 1
        total += int(1 / (end_time - start_time))
        # print('recode FPS:' + str(total // cnt))

def task2(array):
    model = YOLO('./model/yolov8n-pose.pt')  # load a custom model
    data = frombuffer(array, dtype=numpy.uint8, count=len(array))
    data = data.reshape((1440,2560,3))
    while True:
        start_time = time.time()
        image = model(data)[0].plot()
        end_time = time.time()
        cv2.putText(image, 'fps:' + str(1 // (end_time - start_time)), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.namedWindow('screen',0)
        cv2.resizeWindow('screen', 1980, 1080)
        cv2.imshow('screen', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
 
if __name__ == '__main__':
    n = 1440*2560*3
    array = RawArray('d', n)
    data = frombuffer(array, dtype=int, count=len(array))
    data = data.reshape((1440,2560,3))
    p1 = Process(target=task1, args=(array,))
    p2 = Process(target=task2, args=(array,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    