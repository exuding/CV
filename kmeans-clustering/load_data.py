#_*_coding:utf-8_*_
'''
@project: cv-classification
@author:
@time: 2020/5/21 7:06 下午
'''

import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataSet = FlameSet('./data')
    print(dataSet[2].shape)