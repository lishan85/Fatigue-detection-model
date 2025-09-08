import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ========== 全局变量 ==========
face_model_path = r'D:/yolo_pytorch/yolov12l-face.pt'
model = YOLO(face_model_path)

root = tk.Tk()
root.title("疲劳检测参数调节器")
# 滑块默认值
EAR_VAL   = tk.DoubleVar(value=0.25)
MAR_VAL   = tk.DoubleVar(value=0.8)
PITCH_VAL = tk.DoubleVar(value=20)
ROLL_VAL  = tk.DoubleVar(value=20)

LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387]
MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# ========== 工具函数 ==========
def ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    v1 = pts[1][1] - pts[5][1]
    v2 = pts[2][1] - pts[4][1]
    h_dist = pts[3][0] - pts[0][0] + 1e-6
    return (v1 + v2) / 2 / h_dist

def head_pose(landmarks, w, h):
    model = np.array([(0, 0, 0), (0, -330, -65), (-225, 170, -135),
                      (225, 170, -135), (-150, -150, -125), (150, -150, -125)])
    image = np.array([(landmarks[1].x * w, landmarks[1].y * h),
                      (landmarks[152].x * w, landmarks[152].y * h),
                      (landmarks[33].x * w, landmarks[33].y * h),
                      (landmarks[263].x * w, landmarks[263].y * h),
                      (landmarks[61].x * w, landmarks[61].y * h),
                      (landmarks[291].x * w, landmarks[291].y * h)], dtype="double")
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
    _, rvec, _ = cv2.solvePnP(model, image, cam, np.zeros((4, 1)))
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.arcsin(R[1, 2]) * 57.3
    roll  = np.arctan2(R[1, 0], R[0, 0]) * 57.3
    return pitch, roll

# ========== 主处理线程 ==========
def processing_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mesh = mp_face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                left_ear  = ear(lm, LEFT_EYE,  w, h)
                right_ear = ear(lm, RIGHT_EYE, w, h)
                mouth_mar = ear(lm, MOUTH,      w, h)
                pitch, roll = head_pose(lm, w, h)

                eye_closed = (left_ear < EAR_VAL.get()) or (right_ear < EAR_VAL.get())
                yawn       = mouth_mar > MAR_VAL.get()
                nod        = abs(pitch) > PITCH_VAL.get() or abs(roll) > ROLL_VAL.get()

                for flag, idx, color, label in [
                    (eye_closed, LEFT_EYE,  (0,0,255) if eye_closed else (0,255,0), "EYE"),
                    (yawn,       MOUTH,     (0,0,255) if yawn else (255,0,0),     "MOUTH")
                ]:
                    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx]
                    x_min, y_min = min(p[0] for p in pts), min(p[1] for p in pts)
                    x_max, y_max = max(p[0] for p in pts), max(p[1] for p in pts)
                    cv2.rectangle(crop, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(crop, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 实时文字
                cv2.putText(display, f"EAR:{left_ear:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, f"MAR:{mouth_mar:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(display, f"Pitch:{pitch:.1f} Roll:{roll:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if eye_closed or yawn or nod:
                    cv2.putText(display, "WARNING: FATIGUE!", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            display[y1:y2, x1:x2] = crop

        cv2.imshow("Camera Eye Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ========== GUI ==========

tk.Label(root, text="EAR阈值").grid(row=0, column=0, padx=5, pady=5)
tk.Scale(root, variable=EAR_VAL, from_=0.1, to=0.5, resolution=0.01, orient=tk.HORIZONTAL).grid(row=0, column=1)

tk.Label(root, text="MAR阈值").grid(row=1, column=0, padx=5, pady=5)
tk.Scale(root, variable=MAR_VAL, from_=0.4, to=1.2, resolution=0.05, orient=tk.HORIZONTAL).grid(row=1, column=1)

tk.Label(root, text="Pitch阈值").grid(row=2, column=0, padx=5, pady=5)
tk.Scale(root, variable=PITCH_VAL, from_=10, to=40, resolution=1, orient=tk.HORIZONTAL).grid(row=2, column=1)

tk.Label(root, text="Roll阈值").grid(row=3, column=0, padx=5, pady=5)
tk.Scale(root, variable=ROLL_VAL, from_=10, to=40, resolution=1, orient=tk.HORIZONTAL).grid(row=3, column=1)

tk.Button(root, text="开始检测", command=lambda: root.after(100, processing_loop)).grid(row=4, column=0, columnspan=2, pady=10)
root.mainloop()