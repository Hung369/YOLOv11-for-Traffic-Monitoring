from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    results = model.train(
        data="./data_training/data.yaml", epochs=300, imgsz=640, device=0, save=True,
        batch=64, lr0=0.01, momentum=0.937, weight_decay=0.0005,  cos_lr=True
    )