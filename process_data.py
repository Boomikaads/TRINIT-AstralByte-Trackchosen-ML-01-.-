import os
import xml.etree.ElementTree as ET
import cv2

# Function to parse XML annotation files
def parse_annotation(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'name': obj_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return filename, image_width, image_height, objects

# Path to the directory containing annotation files
annotations_dir = r'/Users/boomika/Desktop/xmls'

# Path to the directory containing images
images_dir = r'/Users/boomika/Desktop/images'

# List to store parsed annotations
annotations = []

# Parse each annotation file in the directory
for annotation_file in os.listdir(annotations_dir):
    annotation_path = os.path.join(annotations_dir, annotation_file)
    filename, image_width, image_height, objects = parse_annotation(annotation_path)
    image_path = os.path.join(images_dir, filename)
    annotations.append({
        'image_path': image_path,
        'image_width': image_width,
        'image_height': image_height,
        'objects': objects
    })

# Load images and corresponding annotations
for annotation in annotations:
    image = cv2.imread(annotation['image_path'])
    
    # Draw bounding boxes on the image
    for obj in annotation['objects']:
        xmin, ymin, xmax, ymax = obj['bbox']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()