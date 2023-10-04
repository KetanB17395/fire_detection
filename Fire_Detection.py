
from ultralytics import YOLO
from pandas import pandas
model = YOLO("best117e.pt")
results = model.predict(source="/home/ketan/fire/YoloV8 model/fire_-20230425T100341Z-001/fire_/vlc-record-2023-04-25-13h49m54s-rtsp___161.9.5.205_554_cam_realmonitor-.mp4", show=True)
breakpoint()
confidence=results[0].boxes.conf.cpu().numpy()
print(confidence)
#results = model.predict(source="0", show=True)
#results = model.predict(source="image.jpg", show=True)
