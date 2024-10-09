import os
from PIL import Image
import cv2
import numpy as np

# Example usage
image_folder = '../Data/valid/images'           # Folder containing original images
annotation_folder = '../Data/valid/labels'  # Folder containing corresponding .txt files
output_folder = '../Cropped_Data/valid/'           # Folder to save the new images

classes= ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic container', 'styrofoam', 'tire']
class_danger=[1,1,3,3,0,0,2]

max_area=0
dimensions=(0,0)


def crop_and_center_bounding_boxes(image_folder, annotation_folder, output_folder):
    global dimensions
    global max_area
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all images in the image folder
    for image_file in os.listdir(image_folder):
        # Skip files that are not images
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Construct full path to the image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # Construct the full path to the annotation file
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')

        # Check if the annotation file exists
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file {annotation_path} does not exist. Skipping.")
            continue

        count=0

        # Read the annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Iterate through each line in the annotation file
        for line in lines:
            count+=1
            # Parse the bounding box information
            class_id, x_center, y_center, width, height = map(float, line.split())
            class_id=int(class_id)

            if class_id !=5 and class_id !=4:
                danger=class_danger[class_id]
            elif class_id==4:
                area=width*height
                if area<0.0006479:
                    danger=1
                elif area<0.0014886:
                    danger=2
                else:
                    danger=3
            elif class_id==5:
                area=width*height
                if area<0.004509:
                    danger=2
                else:
                    danger=3
            else:
                danger=0


            # Calculate the bounding box coordinates
            x_center = x_center * image.width
            y_center = y_center * image.height
            width = width * image.width
            height = height * image.height
            area=width*height

            if max_area<area:
                max_area=area
                dimensions=(width,height)
            
            # Calculate the box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop the image
            crop = image.crop((x1, y1, x2, y2))
            crop=np.array(crop)



            if crop.shape[0] > 400 or crop.shape[1] > 400:
                scale = min(400 / crop.shape[0], 400 / crop.shape[1])
                new_width = int(crop.shape[1] * scale)
                new_height = int(crop.shape[0] * scale)
                crop = cv2.resize(crop, (new_width, new_height))

            # Create a new 400x400 black image
            black_image = np.zeros((400, 400, 3), dtype=np.uint8)

            # Calculate the position to center the cropped image
            y_offset = (400 - crop.shape[0]) // 2
            x_offset = (400 - crop.shape[1]) // 2

            # Place the cropped image in the center of the black image
            black_image[y_offset:y_offset + crop.shape[0], x_offset:x_offset + crop.shape[1]] = crop
            black_image = Image.fromarray(black_image) 
            # Save the new image
            output_image_file = os.path.join(output_folder, f"{danger}/{os.path.splitext(image_file)[0]}_class_{int(class_id)}_{count}.png")
            black_image.save(output_image_file)
            print(f"Saved: {output_image_file}")



crop_and_center_bounding_boxes(image_folder, annotation_folder, output_folder)
print(max_area)
print(dimensions)