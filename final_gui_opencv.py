import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

# ========== 全局变量 ==========
face_model_path = r'D:/yolo_pytorch/yolov12l-face.pt'   # 改成你的
model = YOLO(face_model_path)

root = tk.Tk()
root.title("疲劳检测实时调参")

# 默认阈值
EAR_TH   = tk.DoubleVar(value=0.20)
MAR_TH   = tk.DoubleVar(value=0.80)
PITCH_TH = tk.DoubleVar(value=20)
ROLL_TH  = tk.DoubleVar(value=20)

# 计数器
eye_closed_cnt = 0
yawn_cnt       = 0
nod_cnt        = 0
last_reset     = time.time()

# 关键点
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387]
MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
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
    if w < 30 or h < 30:
        return 0, 0
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
    roll = np.arctan2(R[1, 0], R[0, 0]) * 57.3
    return pitch, roll

# ========== GUI 主窗口 ==========

root.geometry("1100x500")   # 总窗口大小

# 左侧：视频
video_label = tk.Label(root)
video_label.pack(side="left", padx=10, pady=10)

# 右侧：控制面板
ctrl = tk.Frame(root)
ctrl.pack(side="right", padx=10, pady=10)

tk.Label(ctrl, text="EAR阈值").pack()
tk.Scale(ctrl, variable=EAR_TH, from_=0.1, to=0.5, resolution=0.01,
         orient=tk.HORIZONTAL, length=200).pack()

tk.Label(ctrl, text="MAR阈值").pack()
tk.Scale(ctrl, variable=MAR_TH, from_=0.4, to=1.2, resolution=0.05,
         orient=tk.HORIZONTAL, length=200).pack()

tk.Label(ctrl, text="Pitch阈值").pack()
tk.Scale(ctrl, variable=PITCH_TH, from_=10, to=40, resolution=1,
         orient=tk.HORIZONTAL, length=200).pack()

tk.Label(ctrl, text="Roll阈值").pack()
tk.Scale(ctrl, variable=ROLL_TH, from_=10, to=40, resolution=1,
         orient=tk.HORIZONTAL, length=200).pack()

# ========== 视频更新线程 ==========
def update_frame():
    global eye_closed_cnt, yawn_cnt, nod_cnt, last_reset

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

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

            eye_closed = (left_ear < EAR_TH.get()) or (right_ear < EAR_TH.get())
            yawn       = mouth_mar > MAR_TH.get()
            nod        = abs(pitch) > PITCH_TH.get() or abs(roll) > ROLL_TH.get()

            # 计数
            if eye_closed: eye_closed_cnt += 1
            if yawn:       yawn_cnt       += 1
            if nod:        nod_cnt        += 1

            # 每 10 秒评估
            if time.time() - last_reset >= 10:
                fatigue_score = (eye_closed_cnt > 30) + (yawn_cnt > 3) + (nod_cnt > 2)
                if fatigue_score >= 2:
                    cv2.putText(display, "WARNING: FATIGUE!", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                eye_closed_cnt = yawn_cnt = nod_cnt = 0
                last_reset = time.time()

            # 画框
            for flag, idx, label, color in [
                (eye_closed, LEFT_EYE,  'L-EYE', (0,0,255) if flag else (0,255,0)),
                (eye_closed, RIGHT_EYE, 'R-EYE', (0,0,255) if flag else (0,255,0)),
                (yawn,       MOUTH,     'MOUTH', (0,0,255) if flag else (255,0,0))
            ]:
                pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx]
                x_min, y_min = min(p[0] for p in pts), min(p[1] for p in pts)
                x_max, y_max = max(p[0] for p in pts), max(p[1] for p in pts)
                cv2.rectangle(crop, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), color, 2)
                cv2.putText(crop, label, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            display[y1:y2, x1:y2] = crop

    img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imencode('.ppm', img)[1].tobytes()
    video_label.configure(image=tk.PhotoImage(data=img))
    root.after(30, update_frame)   # 30 ms 刷新

root.after(30, update_frame)
root.mainloop()