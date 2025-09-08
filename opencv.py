import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

#--------------------------------------配置参数
face_model_path = r'D:/yolo_pytorch/yolov12l-face.pt'
EAR_THRESHOLD = 0.25   #眼睛判断阈值
MAR_THRESHOLD = 0.4   #嘴巴判断阈值
HEAD_PITCH_TH = 20   #低头抬头度数
HEAD_ROLL_TH = 20   #左右歪头度数
WINDOW_SEC = 10   #判断周期

eye_closed_dur = 0
yawn_cnt = 0
nod_cnt = 0
fatigue_warning = False
last_reset = time.time()

#mediaPipe 模型初始化
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

#加载模型
face_model = YOLO(face_model_path)

#--------------------------------------函数
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]   #左眼关键点
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]   #右眼关键点

#landmarks: Mediapipe 返回的 468 个关键点
#eye_indices: 左眼或右眼的6 个点
#w, h: 人脸裁剪图的宽高，把归一化坐标变成像素坐标
def compute_ear(landmarks, eye_indices, w, h):

    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]#把 6 个点的归一化坐标 (0~1) 转成真正的像素坐标

    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))   #上眼皮到下眼皮距离
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))   #上眼皮到下眼皮距离
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))   #眼睛水平长度
    return (A + B) / (2.0 * C + 1e-6)   # EAR = (A+B)/(2C)，眼睛越闭合，EAR 越小

def compute_mar(landmarks, w, h):
    # 取上下嘴唇中心点
    upper_lip_x = (landmarks[13].x + landmarks[14].x) / 2 * w
    upper_lip_y = (landmarks[13].y + landmarks[14].y) / 2 * h
    lower_lip_x = (landmarks[17].x + landmarks[18].x) / 2 * w
    lower_lip_y = (landmarks[17].y + landmarks[18].y) / 2 * h

    mouth_height = np.linalg.norm([upper_lip_x - lower_lip_x, upper_lip_y - lower_lip_y])

    # 取左右嘴角
    mouth_left_x = landmarks[78].x * w
    mouth_left_y = landmarks[78].y * h
    mouth_right_x = landmarks[308].x * w
    mouth_right_y = landmarks[308].y * h

    mouth_width = np.linalg.norm([mouth_left_x - mouth_right_x, mouth_left_y - mouth_right_y])
    # MAR = 高度/宽度，打哈欠时高度变大，MAR 变大
    return mouth_height / (mouth_width + 1e-6)

#计算头部姿势
def head_pose(landmarks, w, h):

    # 3D 模型点（鼻尖、下巴、左右眼、左右嘴角）
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")

    # 对应 2D 图像点
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype="double")

    # 相机内参矩阵
    camera_matrix = np.array([[w, 0, w / 2],
                              [0, w, h / 2],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # solvePnP：用 3D-2D 对应点求解头部旋转
    _, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # 把旋转向量变成旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 从旋转矩阵计算 Pitch（俯仰）和 Roll（侧倾）
    pitch = np.arcsin(R[1, 2]) * 57.3
    roll = np.arctan2(R[1, 0], R[0, 0]) * 57.3
    return pitch, roll

#--------------------------------------主循环
cap = cv2.VideoCapture(0)    #0=默认电脑摄像头
cv2.namedWindow("Eye Detection", cv2.WINDOW_NORMAL)  #创建一个可拉伸的窗口

#无限循环以逐帧处理
while True:
    ret, frame = cap.read()
    if not ret:
        break
    display_frame = frame.copy()

    #人脸检测，置信度大于0.4
    face_results = face_model.predict(source=frame, conf=0.4, verbose=False)[0]

    #遍历每张检测到的人脸框
    for box in face_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])   #把框的坐标转成整数
        face_crop = frame[y1:y2, x1:x2]   #抠出人脸小图
        h, w = face_crop.shape[:2]

        #找关键点
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mesh = mp_face_mesh.process(rgb_crop)

        #拿到点
        if mesh.multi_face_landmarks:
            lm = mesh.multi_face_landmarks[0].landmark   #取脸

            #算分数
            left_ear = compute_ear(lm, LEFT_EYE_IDXS, w, h)
            right_ear = compute_ear(lm, RIGHT_EYE_IDXS, w, h)
            mar = compute_mar(lm, w, h)
            pitch, roll = head_pose(lm, w, h)

            #疲劳判断
            head_tilt = max(abs(pitch), abs(roll))
            is_tilted = head_tilt > max(HEAD_PITCH_TH, HEAD_ROLL_TH)
            eye_closed = left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD
            mouth_open = mar > MAR_THRESHOLD
            head_nod = abs(pitch) > HEAD_PITCH_TH or abs(roll) > HEAD_ROLL_TH

            print(f"left_ear: {left_ear:.3f}, right_ear: {right_ear:.3f}, MAR: {mar:.3f}, pitch: {pitch:.1f}, roll: {roll:.1f}")

            #设置颜色
            eye_status = "OPEN"
            eye_color = (0, 255, 0)
            if eye_closed:
                eye_status = "CLOSED"
                eye_color = (0, 0, 255)
            if is_tilted:
                eye_status += " + TILT"
                eye_color = (0, 255, 255)

            #计数器累加
            if eye_closed: eye_closed_dur += 1
            if mouth_open: yawn_cnt += 1
            if head_nod:   nod_cnt += 1

            #10秒评估一次
            t_now = time.time()
            if t_now - last_reset >= WINDOW_SEC:
                #疲劳分数：闭眼帧数>30 或 打哈欠>3 或 点头>2 各算 1 分
                fatigue_score = (eye_closed_dur > 30) + (yawn_cnt > 3) + (nod_cnt > 2)
                fatigue_warning = fatigue_score >= 2
                eye_closed_dur = yawn_cnt = nod_cnt = 0
                last_reset = t_now

            if fatigue_warning:
                cv2.putText(display_frame, "WARNING",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            #眼睛框
            for eye_idx, label in [(LEFT_EYE_IDXS, 'L-EYE'), (RIGHT_EYE_IDXS, 'R-EYE')]:
                pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_idx]
                x_min, y_min = min(p[0] for p in pts), min(p[1] for p in pts)
                x_max, y_max = max(p[0] for p in pts), max(p[1] for p in pts)
                cv2.rectangle(face_crop, (x_min-3, y_min-3), (x_max+3, y_max+3), eye_color, 2)
                cv2.putText(face_crop, f"{label}:{eye_status}", (x_min, y_min - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)

            #嘴巴框
            mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in [78, 308]]
            x_min, y_min = min(p[0] for p in mouth_pts), min(p[1] for p in mouth_pts)
            x_max, y_max = max(p[0] for p in mouth_pts), max(p[1] for p in mouth_pts)
            mouth_color = (0, 0, 255) if mouth_open else (255, 0, 0)
            cv2.rectangle(face_crop, (x_min-10, y_min-20), (x_max+10, y_max+20), mouth_color, 2)
            cv2.putText(face_crop, f"MOUTH: {'OPEN' if mouth_open else 'CLOSED'}",
                        (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 1)

            #显示头部姿势
            cv2.putText(face_crop, f"Pitch: {pitch:.1f} Roll: {roll:.1f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        display_frame[y1:y2, x1:x2] = face_crop

    cv2.imshow("Eye Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
