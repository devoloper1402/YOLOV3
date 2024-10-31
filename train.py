import config
import torch
import torch.optim as optim

from model import YOLO_V3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fun, scaler, scaled_anchor):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        Y0, Y1, Y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = (
                loss_fun(out[0], Y0, scaled_anchor[0])
                + loss_fun(out[1], Y1, scaled_anchor[1])
                + loss_fun(out[2], Y2, scaled_anchor[2])
            )
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss = mean_loss)

def main():
    model = YOLO_V3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)

    loss_fun = YoloLoss()
    scaler = torch.amp.GradScaler('cuda')

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path = config.DATASET +"/train.csv", test_csv_path=config.DATASET+"/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(test_loader,model,optimizer,loss_fun,scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer)

        if epoch > 0 and epoch % 10 == 0:
            print("On Test Loader: ")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

if __name__ == "__main__":
    main()