import cv2
import numpy as np
import os

classes= ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic container', 'styrofoam', 'tire']
class_danger=[1,1,3,3,0,0,2]


def process_images(image_dir, label_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all images in the image directory
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        # Read the image
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        # Construct the corresponding label file path
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        # Read bounding box information
        with open(label_path, 'r') as f:
            boxes = f.readlines()

        for i, box in enumerate(boxes):
            # Parse bounding box information
            class_id, x_center, y_center, width, height = map(float, box.split())
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


            # Convert to integer
            x_center, y_center, width, height = int(x_center), int(y_center), int(width), int(height)

            # Calculate the bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop the bounding box from the image
            crop = image[y1:y2, x1:x2]

            # Scale down if the bounding box is greater than 400x400
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

            # Save the new image
            output_image_name = f"{os.path.splitext(image_name)[0]}_box{i}.png"
            output_image_path = os.path.join(output_dir,str(danger), output_image_name)
            cv2.imwrite(output_image_path, black_image)

# Define your directories
image_directory = 'Data/train/images'    # Directory containing images
label_directory = 'Data/train/labels'   # Directory containing text files with bounding box info
output_directory = 'Cropped_Data/train/'   # Directory where cropped images will be saved

# Process the images
process_images(image_directory, label_directory, output_directory)