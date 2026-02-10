import json
import os
import time
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from src import model
from src.model import SimpleDetector
from src.detection_dataset import DetectionDataset

def split_dataset(json_file, test_size=0.33, random_state=42):
    with open(json_file, "r") as f:
        data = json.load(f)

    labels = [item["class_id"] for item in data]
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=labels)
    return train, test


def train_network(train_loader, epochs):
    resnet = resnet50(pretrained=True)

    # Freeze the pre-trained layers
    for param in resnet.parameters():
        param.requires_grad = False
         
    model = SimpleDetector(resnet, num_classes=38)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classLossFunc = nn.CrossEntropyLoss()
    bboxLossFunc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        running_loss = 0.0

        for images, boxes, labels in train_loader:
            images = images.to(device)
            boxes  = boxes.to(device)
            labels = labels.to(device)

            predictions = model(images) 
            bboxLoss = bboxLossFunc(predictions[0], boxes)
            classLoss = classLossFunc(predictions[1], labels)
            loss = bboxLoss + classLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

    torch.save({
    "model_state": model.state_dict(),
    }, "model.pth")

def evaluate_network(test_loader):
    resnet = resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
         
    model = SimpleDetector(resnet, num_classes=38)
    model.load_state_dict(torch.load("model.pth")["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_gts   = []
    all_images = []

    with torch.no_grad():
        for images, boxes, labels in test_loader:
            images = images.to(device)
            boxes  = boxes.to(device)

            box_preds, class_preds = model(images)
            all_preds.append(box_preds.cpu())
            all_gts.append(boxes.cpu())
            all_images.append(images.cpu())

    return all_preds, all_gts, all_images

def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box

    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    return x1, y1, x2, y2
        
def draw_detection(pred, gt, image):
    w,h = 256, 256
    gt_xyxy   = yolo_to_xyxy(gt, w, h)
    pred_xyxy = yolo_to_xyxy(pred, w, h)

    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1,2,0))

    # GT box (green)
    x1,y1,x2,y2 = gt_xyxy
    ax.add_patch(patches.Rectangle(
        (x1,y1), x2-x1, y2-y1,
        edgecolor='g', linewidth=2, fill=False
    ))

    # Pred box (red)
    x1,y1,x2,y2 = pred_xyxy
    ax.add_patch(patches.Rectangle(
        (x1,y1), x2-x1, y2-y1,
        edgecolor='r', linewidth=2, fill=False
    ))

    plt.show()

train, test = split_dataset("dataset.json")
train_dataset = DetectionDataset(train, max_samples=500)
test_dataset  = DetectionDataset(test, max_samples=500)

size = 8
trainLoader = DataLoader(train_dataset, batch_size=size, shuffle=True, num_workers=os.cpu_count())
testLoader = DataLoader(test_dataset, batch_size=size, num_workers=os.cpu_count())
# train_network(trainLoader, epochs=100)
preds, gts, images = evaluate_network(testLoader)
preds = torch.cat(preds, dim=0)
gts = torch.cat(gts, dim=0)
print(len(images))

# detection#
i = 4
j = 1
k = j * size + i
draw_detection(preds[k], gts[k], images[j][i])

