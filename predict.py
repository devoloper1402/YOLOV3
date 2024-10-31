import torch
import cv2
import numpy as np
from model import YOLO_V3
from utils import non_max_suppression, cells_to_bboxes
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model and set to evaluation mode
def load_model():
    checkpoint = torch.load("my_checkpoint.pth.tar", map_location=config.DEVICE, weights_only=True)
    model = YOLO_V3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# Function to predict on an image
def predict_image(image_path, model, anchors, threshold=0.4):
    # Preprocess the image
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image / 255.0  # normalize to [0,1]
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)

    # Perform prediction
    with torch.no_grad():
        predictions = model(image)

    # Process predictions to bounding boxes
    bboxes = []
    for i in range(len(predictions)):
        batch_bboxes = cells_to_bboxes(predictions[i], anchors[i], S=predictions[i].shape[2])
        bboxes += batch_bboxes[0]  # get predictions for the first image in batch

    # Apply non-max suppression to filter boxes
    nms_bboxes = non_max_suppression(bboxes, iou_threshold=config.NMS_IOU_THRESH, threshold=threshold, box_format="midpoint")

    # Scale bounding boxes to original image dimensions
    for box in nms_bboxes:
        box[1] *= orig_w  # scale x-center
        box[2] *= orig_h  # scale y-center
        box[3] *= orig_w  # scale width
        box[4] *= orig_h  # scale height

    return nms_bboxes

# Function to plot bounding boxes
def plot_boxes(image_path, bboxes):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in bboxes:
        class_pred = int(box[0])
        x, y, w, h = box[1], box[2], box[3], box[4]
        rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        plt.text(x - w / 2, y - h / 2, f"Class: {class_pred}", color="white", fontsize=8, backgroundcolor="black")

    plt.show()

if __name__ == "__main__":
    model = load_model()
    image_path = "C://Users//tgmad//PycharmProjects//YoloScratch//PASCAL_VOC//images//000001.jpg"  # Replace with your image file
    anchors = torch.tensor(config.ANCHORS, device=config.DEVICE) * torch.tensor(config.S, device=config.DEVICE).unsqueeze(1).unsqueeze(2)

    bboxes = predict_image(image_path, model, anchors)
    print("Detected Boxes:", bboxes)

    plot_boxes(image_path, bboxes)
