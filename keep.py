
import os
import os.path

import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
from torchvision.utils import save_image
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, path):
        self.name = 'COCO'
        self.coco = COCO(path)
        self.ids = list(self.coco.imgs.keys())

    def _get_image_path(self, file_name):
        return os.path.join('http://images.cocodataset.org/val2017', file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']

        response = requests.get(self._get_image_path(file_name))
        img = Image.open(BytesIO(response.content)).convert('RGB')

        transform = transforms.ToTensor()

        # Imageをテンソルに変換
        img = transform(img)

        return img, img_id

    def __len__(self):
        return len(self.ids)


def main():
    def crop_image(self, img, bbox):
        x_min, y_min, width, height = map(int, bbox)

        # 画像をクロップ
        cropped_img = transforms.functional.crop(
            img, x_min, y_min, height, width
        )
        # print(bbox)
        # print(cropped_img.size())

        return cropped_img

    path = '/mnt/NAS-TVS872XT/dataset/COCO/annotations/person_keypoints_val2017.json'

    # データセットのインスタンス化
    dataset = CocoDataset(path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    # DataLoaderを使用してデータをループする例
    for idx, (img, anns) in enumerate(dataloader):
        for i in range(len(anns)):
            cropped_img = crop_image(img, anns[i]['bbox'])
            save_image(img, f'test/def/image{idx}.png')
            save_image(cropped_img, f'test/crop/cropped_image{idx}.png')

        print(idx)

        # img = np.array(img)


if __name__ == '__main__':
    main()
