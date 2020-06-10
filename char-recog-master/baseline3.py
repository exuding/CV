#_*_coding:utf-8_*_
'''
@project: char-recog
@author: exudingtao
@time: 2020/5/16 10:46 下午
'''


import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


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

        # 设置最长的字符长度为4个
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (4 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:4]))

    def __len__(self):
        return len(self.img_path)


class Gauss_blur():
    def __init__(self, w, std):
        self.w = w
        self.std = std

    def __call__(self, img):
        if random.randint(0, 1) == 0:
            return img
        else:
            img = np.asarray(img, np.float)
            img = cv2.GaussianBlur(img, (self.w, self.w), self.std)
            img = Image.fromarray(np.uint8(img))
            return img



class Noise():
    """Adds gaussian noise to a tensor.

    Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     Noise(0.1, 0.05)),
        >>> ])

    """

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        if random.randint(0, 1) == 0:
            return tensor
        else:
            noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
            return tensor.add_(noise)

model_urls = {'resnet50': './weights/resnet50.pth',}

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet50(pretrained=False)
        pre = torch.load(model_urls['resnet50'])
        model_conv.load_state_dict(pre)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        self.fc1 = nn.Linear(2048, 11)
        self.fc1_2 = nn.Linear(1024, 11)
        self.fc2 = nn.Linear(2048, 11)
        self.fc2_2 = nn.Linear(1024, 11)
        self.fc3 = nn.Linear(2048, 11)
        self.fc3_2 = nn.Linear(1024, 11)
        self.fc4 = nn.Linear(2048, 11)
        self.fc4_2 = nn.Linear(1024, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4


def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0, c1, c2, c3 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3])

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(loss.item())

        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3 = model(input)
                output = np.concatenate([
                    c0.data.numpy(),
                    c1.data.numpy(),
                    c2.data.numpy(),
                    c3.data.numpy()], axis=1)
                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta




if __name__ == '__main__':

    train_path = glob.glob('./data/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('./data/mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    print(len(train_path), len(train_label))

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((70, 140)),
                        transforms.RandomCrop((60, 120)),
                        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                        #添加的
                        transforms.RandomGrayscale(0.5),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        Noise(0, 0.05),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=True,
        num_workers=8,
    )

    val_path = glob.glob('./data/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('./data/mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    print(len(val_path), len(val_label))

    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=8,
    )


    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    best_loss = 1000.0

    use_cuda = False
    if use_cuda:
        model = model.cuda()

    for epoch in range(10):

        train_loss = train(train_loader, model, criterion, optimizer)
        val_loss = validate(val_loader, model, criterion)
        print('暂时结束。。。。。')
        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
        ]).T
        print('val_predict_label', val_predict_label)
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print(val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './model.pt')

    test_path = glob.glob('./data/mchar_test_a/*.png')
    test_path.sort()
    test_label = [[1]] * len(test_path)
    print(len(val_path), len(val_label))


    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.RandomCrop((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=5,
    )

    test_predict_label = predict(test_loader, model, 1)

    test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    test_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
    ]).T

    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))

    import pandas as pd

    df_submit = pd.read_csv('./data/mchar_sample_submit_A.csv')
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv('renset18_4.csv', index=None)