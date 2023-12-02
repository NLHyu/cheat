from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./model/yolov8n-pose.pt')
source = 'screen'
# Define current screenshot as source
while True:
    

    # Run inference on the source
    results = model(source)  # list of Results objects
    # print(results[0].plot())