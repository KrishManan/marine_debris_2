from ultralytics import YOLO
if __name__=="__main__":
    model = YOLO("yolov8l.pt")

    model.train(data="../Data/data.yaml", epochs=1)