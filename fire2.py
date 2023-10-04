import cv2
import numpy as np
from PIL import Image

net = cv2.dnn.readNetFromONNX('/home/ketan/fire/gevbest.onnx')

INPUT_WIDTH = 800
INPUT_HEIGHT = 800
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.3
Numberc = ['Fire']

cap = cv2.VideoCapture('video_data_fire/testing_fire.mp4') 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    preds = preds.transpose((0, 2, 1))
    class_ids, confs, boxes = list(), list(), list()
    
    image_height, image_width, _ = frame.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    
    rows = preds[0].shape[0]
    
    for i in range(rows):
        row = preds[0][i]
        conf = row[4]
        
        classes_score = row[4:]
        _, _, _, max_idx = cv2.minMaxLoc(classes_score)
        class_id = max_idx[1]
        
        if classes_score[class_id] > CONFIDENCE_THRESHOLD:
            confs.append(conf)
            label = Numberc[int(class_id)]
            class_ids.append(label)
            
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)
    
    r_class_ids, r_confs, r_boxes = list(), list(), list()
    
    indexes = cv2.dnn.NMSBoxes(boxes, confs, SCORE_THRESHOLD, NMS_THRESHOLD)
    
    for i in indexes:
        r_class_ids.append(class_ids[i])
        r_confs.append(confs[i])
        r_boxes.append(boxes[i])
        
    for i in indexes:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
    
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)
        #if i < len(r_class_ids) and i < len(r_confs):
        #label = f'{r_class_ids[i]}: {r_confs[i]:.2f}'
        cv2.putText(frame, str(r_confs), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        #cv2.imshow('Modified Image', frame)
        #cv2.waitKey(0)
        #output_path = 'path/to/save/modified_image.jpg'
        #cv2.imwrite(output, frame)
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('Video Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
