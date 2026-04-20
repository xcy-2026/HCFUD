from ultralytics import YOLO
import os

model = YOLO('runs/train_base/exp/weights/best.pt')  # 权重路径
source = 'D:/dataset/biandianzhan_ours/test/images'  # 图片目录
save_dir = 'runs/predict'                             # 保存目录
conf = 0.25                                           # 置信度阈值

os.makedirs(save_dir, exist_ok=True)

model.predict(source, save=True, conf=conf, project=save_dir, name='.', exist_ok=True)
