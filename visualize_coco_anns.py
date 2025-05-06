import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

def visualize_coco_annotations(coco_json_path, image_dir, output_dir=None, num_images=5, random_selection=True):
    """
    Visualize COCO format annotations for NuScenes dataset.
    
    Args:
        coco_json_path: Path to the COCO format JSON file
        image_dir: Base directory where the images are stored
        output_dir: Directory to save visualized images (if None, just display them)
        num_images: Number of images to visualize
        random_selection: If True, randomly select images, otherwise use the first N
    """
    # Load the COCO format JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping from image_id to image filename
    image_id_to_filename = {img['id']: img['absolute_url'] for img in coco_data['images']}
    
    # Create a mapping from category_id to category name
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create a mapping from image_id to annotations
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # Define a color map for different categories
    colors = plt.cm.rainbow(np.linspace(0, 1, len(category_id_to_name)))
    category_id_to_color = {cat_id: colors[i][:3] for i, cat_id in enumerate(category_id_to_name.keys())}
    
    # Select images to visualize
    image_ids = list(image_id_to_filename.keys())
    if random_selection:
        selected_image_ids = random.sample(image_ids, min(num_images, len(image_ids)))
    else:
        selected_image_ids = image_ids[:min(num_images, len(image_ids))]
    
    # Visualize each selected image with its annotations
    for image_id in selected_image_ids:
        filename = image_id_to_filename[image_id]
        
        # Handle absolute paths or adjust path as needed
        if filename.startswith('/'):
            # This is an absolute path, might need to adjust based on your environment
            image_path = filename
        else:
            # This is a relative path
            image_path = os.path.join(image_dir, filename)
        
        try:
            image = Image.open(image_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(image))
            plt.title(f"Image ID: {image_id}")
            
            # Plot bounding boxes
            if image_id in image_id_to_annotations:
                for ann in image_id_to_annotations[image_id]:
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    color = category_id_to_color[category_id]
                    
                    # Create a Rectangle patch
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    
                    # Add the rectangle to the plot
                    plt.gca().add_patch(rect)
                    
                    # Add label
                    category_name = category_id_to_name[category_id]
                    plt.text(
                        bbox[0], bbox[1] - 5, category_name,
                        fontsize=10, color=color, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
                    )
            else:
                print(f"No annotations found for image {image_id}")
            
            plt.axis('off')
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"visualized_{image_id}.png"), 
                           bbox_inches='tight', dpi=150)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")

def main():
    # Replace these paths with your actual paths
    coco_json_path = "data/nuscenes/annotations/mini_nuscenes_2d_val.json"
    image_dir = "/"  # Root directory since absolute paths are used in the JSON
    output_dir = ".out/visualization_results"
    
    # Visualize 5 random images
    visualize_coco_annotations(
        coco_json_path=coco_json_path,
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=100,
        random_selection=True
    )

if __name__ == "__main__":
    main()