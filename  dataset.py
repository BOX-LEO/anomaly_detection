import json
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def json_to_mask(json_data):
    h, w = json_data['imageHeight'], json_data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in json_data['shapes']:
        if shape['label'] == 'defect':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask

def mask_to_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
    return bboxes

def display_image_with_mask(image_path, json_path,output_path):
    image = cv2.imread(image_path)
    json_data = load_json(json_path)
    mask = json_to_mask(json_data)
    # draw contours of the mask on the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    # save the image with mask
    basename = os.path.basename(image_path)
    output_image_path = os.path.join(output_path, basename)
    cv2.imwrite(output_image_path, image)
    


class DefectSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(512, 512)):
        self.normal_dir = os.path.join(root_dir, 'normal')
        self.abnormal_dir = os.path.join(root_dir, 'abnormal')
        self.image_size = image_size
        self.transform = transform

        # List images
        self.normal_images = [f for f in os.listdir(self.normal_dir) if f.endswith(('.jpeg', '.png'))]
        self.abnormal_images = [f for f in os.listdir(self.abnormal_dir) if f.endswith(('.jpeg', '.png')) and not f.endswith('.json')]

        # Combine paths with label flag
        self.samples = [(os.path.join(self.normal_dir, f), None) for f in self.normal_images] + \
                       [(os.path.join(self.abnormal_dir, f), os.path.join(self.abnormal_dir, f.replace('.jpeg', '.json'))) for f in self.abnormal_images]

    def __len__(self):
        return len(self.samples)

    def load_mask_from_json(self, json_path, height, width):
        with open(json_path, 'r') as f:
            data = json.load(f)
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in data['shapes']:
            if shape['label'] == 'defect':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
        return mask

    def __getitem__(self, idx):
        image_path, json_path = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)

        # Load or create mask
        if json_path is None:
            mask = np.zeros(self.image_size, dtype=np.uint8)  # all background
        else:
            original = cv2.imread(image_path)
            h, w = original.shape[:2]
            mask = self.load_mask_from_json(json_path, h, w)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensors (if needed)
        if self.transform:
            image = self.transform(image)

        mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)
        mask = mask.astype(np.float32)

        return image, mask
