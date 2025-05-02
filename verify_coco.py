import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def analyze_coco_dataset(coco_file):
    """
    Analyze COCO dataset statistics
    """
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Extract data
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Count annotations per category
    cat_counts = {}
    for ann in annotations:
        cat_id = ann['category_id']
        cat_name = categories[cat_id]
        cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

    # Count images with annotations
    img_with_anns = set()
    for ann in annotations:
        img_with_anns.add(ann['image_id'])

    # Calculate bbox statistics
    bbox_areas = []
    bbox_aspect_ratios = []

    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        width, height = bbox[2], bbox[3]

        if width > 0 and height > 0:
            area = width * height
            aspect_ratio = width / height

            bbox_areas.append(area)
            bbox_aspect_ratios.append(aspect_ratio)

    # Print statistics
    print(f"Dataset Statistics for {coco_file}")
    print(f"Number of images: {len(images)}")
    print(f"Number of images with annotations: {len(img_with_anns)}")
    print(f"Number of annotations: {len(annotations)}")
    print(f"Number of categories: {len(categories)}")

    print("\nAnnotations per category:")
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name}: {count}")

    # Plot statistics
    plt.figure(figsize=(10, 6))
    plt.bar(cat_counts.keys(), cat_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Annotations per Category')
    plt.tight_layout()
    plt.savefig('category_distribution.png')

    # Plot bbox area histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_areas, bins=50)
    plt.title('Bounding Box Area Distribution')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Count')
    plt.savefig('bbox_area_distribution.png')

    # Plot aspect ratio histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_aspect_ratios, bins=50, range=(0, 5))
    plt.title('Bounding Box Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Count')
    plt.savefig('bbox_aspect_ratio_distribution.png')

    print("\nVisualization saved to:")
    print("  category_distribution.png")
    print("  bbox_area_distribution.png")
    print("  bbox_aspect_ratio_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze COCO dataset statistics')
    parser.add_argument('--coco-file', required=True,
                        help='Path to COCO JSON file')
    args = parser.parse_args()

    analyze_coco_dataset(args.coco_file)
