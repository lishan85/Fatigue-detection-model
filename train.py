import os
from ultralytics import YOLO

data_yaml_path = 'D:/yolo_pytorch/Dataset/data.yaml'

model_name = 'yolo11n.pt'

if __name__ == '__main__':
    model = YOLO(model_name)

    epochs = 100
    batch_size = 16
    imgsz = 640
    name = 'yolo_eye_detection_train'
    device = 0

    print(f"开始训练 {model_name} 模型，使用数据集：{data_yaml_path}")
    print(f"训练轮数: {epochs}, 批次大小: {batch_size}, 图像尺寸: {imgsz}")
    print(f"训练结果将保存到: runs/detect/{name}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=name,
        device=device,
    )

    print("\n训练完成！")
    print(f"训练结果保存在: {model.trainer.save_dir}")
