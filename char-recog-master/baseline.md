# 封装数据和赛题理解


Pytorch 两个处理数据的重要工具类：Dataset和DataLoader

#### Dataset:

任何自定义的数据集类，都要继承来自Pytorch的数据集类，自定义的类必须含有两个函数：__len__(self)和__getitem__(self,idx)

CV问题中Dataset类 需要含有 __getitem__ 和 __len__ 两个方法，__getitem__ 函数需要返回**图片数据**和**图片标签**信息， __len__返回数据大小。

```
#需要继承Dataset类
class ProcessingDataset(Dataset):
    def __inti__(self, ):
        #根据传入的参数进行初始化
        pass
    
    def __getitem__(self, index):
    		#必须实现的函数
        #一般获取图片数据和标签信息，并返回
        return img, label
        
    def __len__(self): 
    		#必须实现的函数
        #返回数据集的大小
        retuen xxx
```

本次比赛的dataset定义：

```
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # 设置最长的字符长度为5个
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)
```



#### Dataloader

DataLoader类位于Pytorch的utils类中，他将数据集对象和不同的采样器联合，提供批量图片。

```

dataloader = torch.utils.data.DataLoader(
		dataset, #上面dataset定义的
    batch_size=40, 
    shuffle=True, 
    num_workers=10,
)
for img,label in dataloader:
	#在数据集上应用深度学习算法
	pass
	
```

本次比赛中的DataLoader设置：

```
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=True, 
    num_workers=10,
)
```

#### 赛题数据

数据标签

对于训练数据每张图片将给出对于的编码标签，和具体的字符框的位置（训练集、测试集和验证集都给出字符位置），可用于模型训练：

| Field  | Description |
| ------ | ----------- |
| top    | 左上角坐标X |
| height | 字符高度    |
| left   | 左上角最表Y |
| width  | 字符宽度    |
| label  | 字符编码    |

字符的坐标具体如下所示：

![image-20200516231817269](https://github.com/thacking/char-recog/blob/master/image/image-20200516231817269.png)

| 原始图片                                                     | 图片JSON标注                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20200516232333544](https://github.com/thacking/char-recog/blob/master/image/1.png) | ![image-20200516232414480](https://github.com/thacking/char-recog/blob/master/image/2.png) |

#### 读取数据

JSON中标签的读取方式

```
import json
train_json = json.load(open('../input/train.json'))

# 数据标注处理
def parse_json(d):
   arr = np.array([
       d['top'], d['height'], d['left'],  d['width'], d['label']
   ])
   arr = arr.astype(int)
   return arr

img = cv2.imread('../input/train/000000.png')
arr = parse_json(train_json['000000.png'])

plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)
plt.imshow(img)
plt.xticks([]); plt.yticks([])

for idx in range(arr.shape[1]):
   plt.subplot(1, arr.shape[1]+1, idx+2)
   plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
   plt.title(arr[4, idx])
   plt.xticks([]); plt.yticks([])
```

![image-20200516232648371](https://github.com/thacking/char-recog/blob/master/image/image-20200516232648371.png)

#### 解题思路

赛题思路分析：赛题本质是分类问题，需要对图片的字符进行识别。但赛题给定的数据图片中不同图片中包含的字符数量不等，如下图所示。有的图片的字符个数为2，有的图片字符个数为3，有的图片字符个数为4。

因此本次赛题的难点是需要对不定长的字符进行识别，与传统的图像分类任务有所不同。

#### 简单定长字符识别：

将赛题抽象为一个定长字符识别问题，在赛题数据集中大部分图像中字符个数为2-4个，最多的字符 个数为6个。
因此可以对于所有的图像都抽象为6个字符的识别问题，字符23填充为23XXXX，字符231填充为231XXX。

经过填充之后，原始的赛题可以简化了6个字符的分类问题。在每个字符的分类中会进行11个类别的分类，假如分类为填充字符，则表明该字符为空。

#### 专业字符识别思路：不定长字符识别

在字符识别研究中，有特定的方法来解决此种不定长的字符识别问题，比较典型的有CRNN字符识别模型。
在本次赛题中给定的图像数据都比较规整，可以视为一个单词或者一个句子。

#### 专业分类思路：检测再识别

在赛题数据中已经给出了训练集、验证集中所有图片中字符的位置，因此可以首先将字符的位置进行识别，利用物体检测的思路完成。此种思路需要参赛选手构建字符检测模型，对测试集中的字符进行识别。选手可以参考物体检测模型SSD或者YOLO来完成。

# 2数据读取与数据扩增

## 数据读取

在Python中有CV领域很多库可以完成数据读取的操作，比较常见的有Pillow和OpenCV。

#### 2.2.1 Pillow

Pillow是Python图像处理函式库(PIL）的一个分支。Pillow提供了常见的图像读取和处理的操作，而且可以与ipython notebook无缝集成，是应用比较广泛的库。

```
from PIL import Image
# 导入Pillow库
# 读取图片
im =Image.open(cat.jpg')
```

```
from PIL import Image, ImageFilter
im = Image.open('cat.jpg')
# 应用模糊滤镜:
im2 = im.filter(ImageFilter.BLUR)
im2.save('blur.jpg', 'jpeg')
```

#### 2.2.2 OpenCV

OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。

```
import cv2
# 导入Opencv库
img = cv2.imread('cat.jpg')
# Opencv默认颜色通道顺序是BRG，转换一下
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

```
import cv2
# 导入Opencv库
img = cv2.imread('cat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 转换为灰度图
```

```
import cv2
# 导入Opencv库
img = cv2.imread('cat.jpg')
img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 转换为灰度图
# Canny边缘检测
edges = cv2.Canny(img, 30, 70)
cv2.imwrite('canny.jpg', edges)
```

## 数据扩增

#### 2.3.1 数据扩增介绍

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。

在深度学习模型的训练过程中，数据扩增是必不可少的环节。现有深度学习的参数非常多，一般的模型可训练的参数量基本上都是万到百万级别，而训练集样本的数量很难有这么多。
其次数据扩增可以扩展样本空间，假设现在的分类模型需要对汽车进行分类，左边的是汽车A，右边为汽车B。如果不使用任何数据扩增方法，深度学习模型会从汽车车头的角度来进行判别，而不是汽车具体的区别。

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别。
对于图像分类，数据扩增一般不会改变标签；对于物体检测，数据扩增不会改变物体坐标位置；对于图像分割，数据扩增不会改变像素标签。

#### 2.3.2 常见的数据扩增方法

在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。

以torchvision为例，常见的数据扩增方法包括：

- transforms.CenterCrop 对图片中心进行裁剪
- transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
- transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
- transforms.Grayscale 对图像进行灰度变换
- transforms.Pad 使用固定值进行像素填充
- transforms.RandomAffine 随机仿射变换
- transforms.RandomCrop 随机区域裁剪
- transforms.RandomHorizontalFlip 随机水平翻转
- transforms.RandomRotation 随机旋转
- transforms.RandomVerticalFlip 随机垂直翻转

在本次赛题中，赛题任务是需要对图像中的字符进行识别，因此对于字符图片并不能进行翻转操作。比如字符6经过水平翻转就变成了字符9，会改变字符原本的含义。

#### 2.3.3 常用的数据扩增库

- #### torchvision

https://github.com/pytorch/vision
pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等；

- #### imgaug

https://github.com/aleju/imgaug
imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快；

- #### albumentations

[https://albumentations.readthedocs.io](https://albumentations.readthedocs.io/)
是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。

## 2.4 Pytorch读取数据

## 2.4 Pytorch读取数据

在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。所以我们只需要重载一下数据读取的逻辑就可以完成数据的读取。

```
import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

train_path = glob.glob('../input/train/*.png')
train_path.sort()
train_json = json.load(open('../input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path, train_label,
          transforms.Compose([
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),

              # 随机颜色变换
              transforms.ColorJitter(0.2, 0.2, 0.2),

              # 加入随机旋转
              transforms.RandomRotation(5),

              # 将图片转换为pytorch 的tesntor
              # transforms.ToTensor(),

              # 对图像像素进行归一化
              # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```

通过上述代码，可以将赛题的图像数据和对应标签进行读取，在读取过程中的进行数据扩增，效果如下所示：

![image-20200516232414480](https://github.com/thacking/char-recog/blob/master/image/3.png)

Dataset：对数据集的封装，提供索引方式的对数据样本进行读取

DataLoder：对Dataset进行封装，提供批量读取的迭代读取

在定义好的Dataset基础上构建DataLoder:

```
import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

train_path = glob.glob('../input/train/*.png')
train_path.sort()
train_json = json.load(open('../input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)),
                       transforms.ColorJitter(0.3, 0.3, 0.2),
                       transforms.RandomRotation(5),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])), 
    batch_size=10, # 每批样本个数
    shuffle=False, # 是否打乱顺序
    num_workers=10, # 读取的线程个数
)

for data in train_loader:
    break
```

在加入DataLoder后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接。此时data的格式为：

```
torch.Size([10, 3, 64, 128]), torch.Size([10, 6])
```

前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签。
