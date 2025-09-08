from ultralytics import YOLO
import cv2

model_path = 'runs/detect/yolo_eye_detection_train3/weights/best.pt'
model = YOLO(model_path)

test_img_path = 'D:/yolo_pytorch/download.png'
img = cv2.imread(test_img_path)

if img is None:
    print("无法加载图片")
    exit()

h, w = img.shape[:2]
scale = 640 / max(h, w)
if scale < 1:
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

results = model.predict(source=img, conf=0.01, iou=0.7, verbose=True)

annotated = results[0].plot()
cv2.imshow("debug", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()