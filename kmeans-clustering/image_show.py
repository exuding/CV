#_*_coding:utf-8_*_
'''
@project: cv-classification
@author:
@time: 2020/5/21 7:05 下午
'''

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


