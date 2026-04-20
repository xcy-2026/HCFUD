import warnings


warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR

if __name__ == '__main__':
    model = YOLO('/data1/lsl/xcy/substation/yolo11n.yaml')

    model.train(data='/data1/lsl/xcy/substation/our_data.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                single_cls=False,  # 是否是单类别检测
                batch=32,
                close_mosaic=10,
                workers=8,
                device='5',
                optimizer='SGD',
                amp=True,
                project='runs/train_base',
                name='exp',
                patience=100,
                # # SID掩码损失配置
                # mask_loss_weight=1,  # 掩码损失总权重（相对于检测损失）
                # mask_alpha=0.1,        # 前景损失权重
                # mask_beta=0.1,         # 背景损失权重
                # mask_gamma=0.05         # 互补性损失权重
                )







