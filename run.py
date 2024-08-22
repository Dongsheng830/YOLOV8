from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("/root/autodl-tmp/YOLOv8/ultralytics/cfg/models/v8/GoldYOLO.yaml")
    # model.load('/root/ultralytics/runs/detect/train3/weights/best.pt')  # 我这里用的n的权重文件，大家可以自行替换自己的版本的
    model.train(data='/root/autodl-tmp/YOLOv8/ultralytics/cfg/datasets/SOD4BD-0.4.yaml',
                cache=False,
                imgsz=[1920, 1080],
                epochs=250,
                batch=4,
                close_mosaic=10,
                workers=8,
                amp=False,  # close amp
                pretrained=False
                )
