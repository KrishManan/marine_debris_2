from ultralytics import YOLO

# Change with your weights file
model = YOLO("../weights/Yolov8best.pt")

# Change with your source file
predictframe=model.predict(source="../src/resources/testimage4.jpg",show=True,save=True)