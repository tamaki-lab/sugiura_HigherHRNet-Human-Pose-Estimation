
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
import pickle
from tqdm import tqdm


class CocoDataset(Dataset):
    def __init__(self, path):
        self.name = 'COCO'
        self.coco = COCO(path)
        self.ids = list(self.coco.imgs.keys())



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]

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
    def crop_image(img, bbox):
        x_min, y_min, width, height = map(int, bbox)

        # 画像をクロップ
        cropped_img = transforms.functional.crop(
            img, y_min, x_min, height, width
        )
        # print(bbox)
        # print(cropped_img.size())

        return cropped_img

    def get_image_path(file_name):
        return os.path.join('http://images.cocodataset.org/val2017', file_name)

    path = '/mnt/NAS-TVS872XT/dataset/COCO/annotations/person_keypoints_val2017.json'

    coco = COCO(path)
    ids = list(coco.imgs.keys())

    for index in tqdm(range(len(ids))):
        img_id = ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']

        response = requests.get(get_image_path(file_name))
        img = Image.open(BytesIO(response.content)).convert('RGB')

        transform = transforms.ToTensor()

        # Imageをテンソルに変換
        img = transform(img)

        # if index == 1:
        #     save_image(img, f'test/image{index}_{i}.png')
        #     save_image(cropped_img, f'test/cropped_image{index}_{i}.png')

        for i in range(len(anns)):
            image_id = anns[i]['image_id']

            os.makedirs(f'test/image/{image_id}/default', exist_ok=True)
            os.makedirs(f'test/image/{image_id}/cropped', exist_ok=True)
            cropped_img = crop_image(img, anns[i]['bbox'])
            if i == 0:
                save_image(img, f'test/image/{image_id}/default/image{i}.png')
            save_image(cropped_img, f'test/image/{image_id}/cropped/image{i}.png')

            os.makedirs(f'test/pickle/{image_id}', exist_ok=True)
            with open(f'test/pickle/{image_id}/data{i}.pkl', 'wb') as f:
                pickle.dump((np.array(img), anns[i]), f)


if __name__ == '__main__':
    main()
