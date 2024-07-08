import json
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import torch


class CocoKeypointsDataset(Dataset):
    def __init__(self, json_file, transform=None):
        """
        Args:
            json_file (string): JSONファイルのパス
            transform (callable, optional): 画像に適用するトランスフォーム
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data['images'])

    def crop_image_with_bbox(self, img, bbox):
        """
        画像をバウンディングボックスでクロップする関数
        Args:
        - img (torch.Tensor): 画像データ
        - bbox (list): バウンディングボックス [x_min, y_min, width, height]
        Returns:
        - cropped_img (torch.Tensor): クロップされた画像
        """
        # バウンディングボックスの座標とサイズを取得し、整数に変換
        x_min, y_min, width, height = map(int, bbox)
        # print(img.size())

        # 画像をクロップ
        cropped_img = transforms.functional.crop(
            img, x_min, y_min, height, width
        )
        # print(bbox)
        # print(cropped_img.size())

        return cropped_img


    def __getitem__(self, idx):
        # 画像データのURLを正しく取得
        img_url = self.data['images'][idx]['coco_url']
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        print(img_url)

        # 変換の定義
        transform = transforms.ToTensor()

        # Imageをテンソルに変換
        img = transform(img)

        num_keypoints = self.data['annotations'][idx]['num_keypoints']
        keypoints = self.data['annotations'][idx]['keypoints']
        bbox = self.data['annotations'][idx]['bbox']
        category_id = self.data['annotations'][idx]['category_id']
        # print(self.data['categories'])

        # 画像をクロップ
        cropped_img = self.crop_image_with_bbox(img, bbox)

        return img, cropped_img, num_keypoints, keypoints, bbox, category_id


path = '/mnt/NAS-TVS872XT/dataset/COCO/annotations/person_keypoints_val2017.json'

# データセットのインスタンス化
dataset = CocoKeypointsDataset(json_file=path)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

# DataLoaderを使用してデータをループする例
for idx, (img, cropped_img, num_keypoints, keypoints, bbox, category_id) in enumerate(dataloader):
    # print(img.size())
    # print(cropped_img.size())
    if category_id.item() == 1:
        if num_keypoints.item() == 0:
            save_image(img, f'0_keypoints/image{idx}.png')
            save_image(cropped_img, f'0_keypoints/ropped_image{idx}.png')
        else:
            save_image(img, f'test/def/image{idx}.png')
            save_image(cropped_img, f'test/crop/cropped_image{idx}.png')
        print(num_keypoints)
    print(idx)

    # img = np.array(img)
