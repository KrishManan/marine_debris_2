import os
import cv2

classes= ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic container', 'styrofoam', 'tire']
class_danger=[]
def crop_bounding_boxes(image_folder, annotation_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images in the image folder
    for image_file in os.listdir(image_folder):
        # Skip files that are not images
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Construct the full path to the image and its corresponding annotation file
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Read the annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Iterate through each line in the annotation file
        for i, line in enumerate(lines):
            # Parse the bounding box information
            class_id, x_center, y_center, width, height = map(float, line.split())
            bbox_area=width*height
            # if class_id:=
            
            # Convert the normalized coordinates to pixel values
            img_height, img_width = image.shape[:2]
            x_center_px = int(x_center * img_width)
            y_center_px = int(y_center * img_height)
            width_px = int(width * img_width)
            height_px = int(height * img_height)

            # Calculate the top-left corner of the bounding box
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            # Crop the image to the bounding box
            cropped_image = image[y1:y2, x1:x2]

            # Save the cropped image
            cropped_image_filename = f"{os.path.splitext(image_file)[0]}_box_{i + 1}.png"
            cv2.imwrite(os.path.join(output_folder, cropped_image_filename), cropped_image)

            print(f"Saved cropped image: {cropped_image_filename}")

# Example usage
image_folder = 'Data/test/images'           # Folder containing original images
annotation_folder = 'Data/test/labels'  # Folder containing corresponding .txt files
output_folder = 'Data/cropped/test'           # Folder to save cropped images

crop_bounding_boxes(image_folder, annotation_folder, output_folder)
