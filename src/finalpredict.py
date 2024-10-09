from ultralytics import YOLO
import cv2 
import numpy as np
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from resnet import resnet50
# from torchvision.models import resnet50

ymodel = YOLO("../Weights/Yolov8nbest.pt") #replace with your stage 1 model path
weights_path = "../Weights/Resnet50best.pth" #replace with your stage 2 model path
img_path="../Examples/Test/testimage.jpg" #replace with your image path



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rmodel = resnet50(num_classes=3).to(device)


assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
rmodel.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))




class_names = ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic con', 'styrofoam', 'tire']

class_indict = ["1", "2", "3"]

def add_text_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(139, 0, 0), thickness=2, bg_color=(255, 255, 255)):
    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the position for the background rectangle
    bg_top_left = (pos[0], pos[1] - text_size[1] - 5)
    bg_bottom_right = (pos[0] + text_size[0], pos[1])
    
    # Draw the filled rectangle behind the text
    cv2.rectangle(img, bg_top_left, bg_bottom_right, bg_color, -1)  # -1 thickness for filling
    
    # Write the text on the image
    cv2.putText(img, text, pos, font, font_scale, color, thickness)


data_transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


img=cv2.imread(img_path)
img=cv2.resize(img,(1000,600))

def main():
    start = time.time()
    results=ymodel.predict(img)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            print(x1, y1, x2, y2)
            label = result.names[int(box.cls)]
            confidence = box.conf.item()


            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

            cropped_h, cropped_w = cropped_img.shape[:2]

            if cropped_h > 400 or cropped_w > 400:
                scale = min(400 / cropped_h, 400 / cropped_w)
                cropped_w = int(cropped_w * scale)
                cropped_h = int(cropped_h * scale)
                cropped_img = cv2.resize(cropped_img, (cropped_w, cropped_h))

            padded_img = np.zeros((400, 400, 3), dtype=np.uint8)
            start_x = (400 - cropped_w) // 2
            start_y = (400 - cropped_h) // 2
            padded_img[start_y:start_y + cropped_h, start_x:start_x + cropped_w] =cropped_img

            input_img = Image.fromarray(padded_img)

            input_img = data_transform(input_img)
            input_img = torch.unsqueeze(input_img, dim=0)

            

            rmodel.eval()
            with torch.no_grad():
                # Predict class
                output = torch.squeeze(rmodel(input_img.to(device))).cpu()

                print(output)
                
                # Compute softmax probabilities
                predict = torch.softmax(output, dim=0)

                print(predict)

                predict_cla = torch.argmax(predict).numpy()

                actual_cla=class_indict[predict_cla]

                print(f"actual class:{actual_cla}")

                # Calculate the weighted average
                weighted_average = (predict[0] * 1 + predict[1] * 2 + predict[2] * 3)
            



            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (139, 0, 0), 4)



                    
            add_text_with_background(img, f'{label} conf:{confidence:.2f} danger:{weighted_average:.2f}', (int(x1), int(y1)-10), bg_color=(255, 255, 255))
    end = time.time()
    timedif = end - start
    print(f"time elapsed:{timedif}")

    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("resources/predictimage.png",img)
    
    

if __name__ == '__main__':
    main()