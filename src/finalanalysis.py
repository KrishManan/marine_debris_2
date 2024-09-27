from ultralytics import YOLO 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from resnet import resnet18
from report import create_report
# from torchvision.models import resnet50

ymodel = YOLO("../weights/Yolov8nbest.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rmodel = resnet18(num_classes=3).to(device)

# load model weights
weights_path = "../Weights/Resnet18best.pth"
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
rmodel.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
width = 2000
height = 1000

print(height,width)

annotated_video_path = 'combined_video.mp4' 
out1=cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))


class_names = ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic con', 'styrofoam', 'tire']
class_mats=['metal','paper','plastic','plastic','plastic','plastic','rubber']
class_indict = ["1", "2", "3"]
danger_classes=['Low(1)','Medium(2)','High(3)']
class_counts = [0] * 7 
danger_counts = [0] * 3
true_dangers=[]
seen_track_ids = set() 
unique_frame_ids = set() 
frame_data = [] 
current_time = 0

colors=['blue', 'dodgerblue', 'royalblue', 'steelblue', 'darkblue', 'deepskyblue', 'aquamarine']
colors2=['deepskyblue','dodgerblue','royalblue']
fig, ax = plt.subplots() 
bars = ax.bar(class_names, class_counts, color=colors)

ax.set_xlabel('Classes') 
ax.set_ylabel('Counts') 
ax.set_title('Class Counts') 
ax.set_ylim(0, 10) 
ax.yaxis.get_major_locator().set_params(integer=True)
plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')

time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12)

fig2, ax2 = plt.subplots() 
bars2 = ax2.bar(class_indict, danger_counts, color=colors2)

ax2.set_xlabel('Danger Levels') 
ax2.set_ylabel('Counts') 
ax2.set_title('Danger Counts') 
ax2.set_ylim(0, 10) 
ax2.yaxis.get_major_locator().set_params(integer=True)


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


def update_bars(frame_data): 
    counts, second = frame_data 
    for bar, count in zip(bars, counts): 
        bar.set_height(count) 
        time_text.set_text(f'Time: {second} s')

def update_bars2(danger_counts): 
    for bar, count in zip(bars2, danger_counts): 
        bar.set_height(count)

data_transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

print(f"Frames per second: {fps}")


framecount = 0 
while cap.isOpened(): 
    success, frame = cap.read()

    if success:
        frame=cv2.resize(frame,(width,height))
        print(f"processing frame {framecount}/{length}")
        results = ymodel.track(frame, persist=True)
        annotated_frame = frame.copy()
        frame_data = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                confidence = box.conf.item()
                id = box.id.numpy()
                id = int(id[-1])
                id = box.id.numpy()
                id = int(id[-1])
                



                cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]

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


                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (139, 0, 0), 4)



                        
                add_text_with_background(frame, f'{label} conf:{confidence:.2f} danger:{weighted_average:.2f}', (int(x1), int(y1)-10), bg_color=(255, 255, 255))
                class_id = int(box.cls)
                track_id = int(box.id)
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    class_counts[class_id] += 1
                    danger_counts[predict_cla] += 1
                    true_dangers.append(weighted_average)
                    if class_counts[class_id] > 10:
                        ax.set_ylim(0, class_counts[class_id])
                    if danger_counts[predict_cla] > 10:
                        ax2.set_ylim(0, danger_counts[predict_cla])
                
            
            

        if framecount % fps == 0 or framecount == length:
            current_time += 1
            frame_data.append((class_counts[:], current_time))
            update_bars(frame_data[0])
            fig.canvas.draw()
            fig.savefig('latest_chart.png')
            plt_img = cv2.imread('latest_chart.png')
            update_bars2(danger_counts)
            fig2.canvas.draw()
            fig2.savefig('latest_chart2.png')
            plt_img2 = cv2.imread('latest_chart2.png')
        
        combined_image = np.zeros((height, width, 3), dtype="uint8")
        annotated_frame = cv2.resize(annotated_frame, (width // 2, height // 2))
        frame= cv2.resize(frame, (width //2, height // 2))  
        plt_img= cv2.resize(plt_img, (width // 2, height // 2))
        plt_img2= cv2.resize(plt_img2, (width // 2, height // 2))
        key=cv2.waitKey(1)
        if key== ord('q'):
            break
        combined_image[:height // 2, :width // 2] = frame
        combined_image[:height // 2, width // 2:width] = annotated_frame
        combined_image[height // 2:height, :width//2] = plt_img
        combined_image[height // 2:height, width // 2:width] = plt_img2

        cv2.imshow("combined image",combined_image)
        cv2.waitKey(1)
        out1.write(combined_image)
        framecount += 1
        create_report("latest_chart.png", "latest_chart2.png", "report.pdf", class_names, class_counts,danger_classes , danger_counts, true_dangers)

    else:     
        print('End of video')
        cap.release()
        break

print(f"Annotated video saved to {annotated_video_path}") 

create_report("latest_chart.png", "latest_chart2.png", "report.pdf", class_names, class_counts,danger_classes , danger_counts, true_dangers)