import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.Sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_NoObj = 10
        self.lambda_Obj = 1
        self.lambda_box = 10


    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        NoObj = target[..., 0] == 0

        #No Object Loss

        No_Object_Loss = self.bce(
            (predictions[..., 0:1][NoObj]), (target[..., 0:1][NoObj]),
        )

        #obejct loss

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_pred = torch.cat([self.Sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        iou = intersection_over_union(box_pred[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (iou * target[..., 0:1][obj]))

        #Box coordinates loss
        predictions[..., 1:3] = self.Sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        #Class Loss
        class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))

        return (self.lambda_class * class_loss + self.lambda_box * box_loss + self.lambda_NoObj * No_Object_Loss + self.lambda_Obj * object_loss )
