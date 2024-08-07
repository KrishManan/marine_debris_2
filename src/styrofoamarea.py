import os

def calculate_and_split_bounding_box_areas(image_folder, annotation_folder):
    # List to store areas of bounding boxes of class 5
    class_5_areas = []

    # Loop through all images in the image folder
    for image_file in os.listdir(image_folder):
        # Skip files that are not images
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Construct the full path to the annotation file
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')

        # Check if the annotation file exists
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file {annotation_path} does not exist. Skipping.")
            continue

        # Read the annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Iterate through each line in the annotation file
        for line in lines:
            # Parse the bounding box information
            class_id, x_center, y_center, width, height = map(float, line.split())
            
            # Check if the class id is 5
            if int(class_id) == 4:
                # Calculate the area of the bounding box
                area = width * height  # Assuming width and height are normalized values
                class_5_areas.append(area)

    # Sort the areas from smallest to greatest
    class_5_areas.sort()

    # Split the areas into 3 equal parts
    n = len(class_5_areas)
    
    # Calculate the size of each part
    part_size = n // 2

    # Create three lists for small, medium, and large areas
    small_areas = class_5_areas[:part_size]
    # medium_areas = class_5_areas[part_size:2*part_size]
    large_areas = class_5_areas[part_size:]

    return small_areas, large_areas

def output_min_max(areas, label):
    if areas:
        print(f"{label} - Min: {min(areas)}, Max: {max(areas)}")
    else:
        print(f"{label} - No areas available.")

# Example usage
image_folder = 'Data/train/images'           # Folder containing original images
annotation_folder = 'Data/train/labels'  # Folder containing corresponding .txt files

small_areas, large_areas = calculate_and_split_bounding_box_areas(image_folder, annotation_folder)

# Output the results
output_min_max(small_areas, "Small Areas")
# output_min_max(medium_areas, "Medium Areas")
output_min_max(large_areas, "Large Areas")
