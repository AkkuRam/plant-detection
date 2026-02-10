import kagglehub
import os
import json

# path = kagglehub.dataset_download("sebastianpalaciob/plantvillage-for-object-detection-yolo")
# print("Path to dataset files:", path)

image_dir = "dataset/images"
label_dir = "dataset/labels"
images = []
labels = []

def get_images_labels(image_dir, label_dir):
    for path in os.listdir(image_dir):
        if path.endswith(".jpg"):
            image_path = os.path.join(image_dir, path)
            label_path = os.path.join(label_dir, path.replace(".jpg", ".txt"))
            images.append(image_path)
            labels.append(label_path)

    return images, labels

def build_dataset(dataset, images, labels):
    for i in range(len(images)):
        boxes = []

        with open(labels[i], "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue 

                class_id = int(parts[0])
                box = list(map(float, parts[1:5]))
                boxes.append(box)

        sample = {
            "image": images[i],
            "class_id": class_id,
            "boxes": boxes
        }
        dataset.append(sample)

    return dataset


images, labels = get_images_labels(image_dir, label_dir)
dataset = build_dataset([], images, labels)

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)