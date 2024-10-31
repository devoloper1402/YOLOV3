import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLO_Dataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir, label_dir,
            anchors,
            S = [13,26,52],
            C=20,
            transform=None
    ):
        self.annotations= pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_threshold = 0.5


    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2),4, axis=1).tolist() #[x,y,w,h, class] so to make it inverted remove np.roll, 4 and axis
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentation = self.transform(image=image, bboxes=bboxes)
            image = augmentation["image"]
            bboxes = augmentation["bboxes"]


            targets = [torch.zeros((self.num_anchors // 3, S,S,6)) for S in self.S]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anch in anchor_indices:
                scale_idx = anch // self.num_anchors_per_scale
                anchor_onscale = anch % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S*y), int(S*x)
                anch_used = targets[scale_idx][anchor_onscale,i,j,0]

                if not anch_used and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_onscale, i, j, 0] = 1
                    Xcell, Ycell = S*x - j, S*y - i
                    wcell, hcell = (
                        width * S,
                        height * S,
                    )
                    box_coordinates = torch.tensor([Xcell, Ycell, wcell, hcell])
                    targets[scale_idx][anchor_onscale,i ,j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_onscale, i,j, 5] =  int(class_label)
                    has_anchor[scale_idx] = True


                elif not anch_used and iou_anchors[anch] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_onscale,i ,j,0] = -1

        return image, tuple(targets)