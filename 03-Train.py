# 移动Arial.ttf到/root/.config/Ultralytics/Arial.ttf
import os
# os.system("cp Arial.ttf /root/.config/Ultralytics/Arial.ttf")
os.system("pip install ultralytics")

os.system("cd /kaggle/working/")
os.chdir("/kaggle/working/")


#延迟启动
import time
time.sleep(30)

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/kaggle/working/Demo_Open/02-YOLODataSet.yaml", epochs=150, imgsz=640, batch=-1)
