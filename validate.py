import sys
import os

# 🔥 强制使用本地的 ultralytics
sys.path.insert(0, '/root/lanyun-tmp/station')

from ultralytics import YOLO

# 加载模型并验证
model = YOLO('runs/train_idea1/exp/weights/best.pt')

metrics = model.val(
    data='/root/lanyun-tmp/station/our_data.yaml',
    batch=32,
    device='0',
    plots=True,
    verbose=True
)

# 打印结果
print("\n" + "="*60)
print("验证结果:")
print("="*60)
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print("="*60)
