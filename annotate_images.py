import os
from ultralytics import YOLO

def create_visdrone_style_annotations(input_dir, output_dir, model_name='yolov8n.pt'):
    """
    Performs object detection on images and saves annotations in VisDrone2019 format.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where annotations will be saved.
        model_name (str): The name of the YOLOv8 model to use (e.g., 'yolov8n.pt').
    """
    # Load the specified YOLOv8 model
    model = YOLO('yolov11n.pt')

    # Create the output annotation directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of image files from the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} images. Starting annotation generation...")

    # Process each image file
    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        
        # Perform inference
        results = model(image_path)
        
        # Define the path for the annotation file
        base_name = os.path.splitext(image_name)[0]
        annotation_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(annotation_path, 'w') as f:
            # Process each detected object in the result
            for box in results[0].boxes:
                # Bounding box coordinates (top-left corner, width, height) in pixels
                xyxy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)
                bbox_left = x1
                bbox_top = y1
                bbox_width = int(x2 - x1)
                bbox_height = int(y2 - y1)
                
                # Confidence score
                score = box.conf[0].item()
                
                # Class ID
                object_category = int(box.cls[0])
                
                # Truncation and Occlusion (default values as YOLO doesn't provide them)
                truncation = 0
                occlusion = 0

                # Write the annotation in VisDrone format
                f.write(
                    f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},"
                    f"{score:.4f},{object_category},{truncation},{occlusion}\n"
                )

        print(f"Generated annotation for {image_name}")

    print(f"\nProcessing complete.")
    print(f"VisDrone-style annotations are saved in: '{output_dir}'")

if __name__ == '__main__':
    # --- Configuration ---
    # Directory containing your images
    INPUT_IMAGE_DIRECTORY = '/home/dev/Yolo11Patch/VisDrone2019-DET-val/images'
    
    # Directory where annotation files will be saved
    OUTPUT_ANNOTATION_DIRECTORY = 'annotations_v11'
    
    # Ensure the input directory exists
    if not os.path.isdir(INPUT_IMAGE_DIRECTORY):
        print(f"Error: Input directory '{INPUT_IMAGE_DIRECTORY}' not found.")
        print("Please create it and add your images.")
    else:
        create_visdrone_style_annotations(INPUT_IMAGE_DIRECTORY, OUTPUT_ANNOTATION_DIRECTORY)