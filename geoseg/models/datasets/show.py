import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        print(filename)
        img = cv2.imread(directory_name + "/" + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(directory_name.replace('image','mask') + "/" + filename)
        mask = (cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        mask =mask/51
        mask[mask[:, :, 0] == 0.0] = [255, 255, 255]
        mask[mask[:, :, 0] == 1.0] = [255, 204, 0]
        mask[mask[:, :, 0] == 2.0] = [0, 0, 255]
        mask[mask[:, :, 0] == 3.0] = [0, 255, 128]
        mask[mask[:, :, 0] == 4.0] = [0, 204, 255]
        mask[mask[:, :, 0] == 5.0] = [0, 0, 0]
        # CLASSES = ('ThinIce', 'ThickIce', 'Sea', 'Land', 'PoolIce', 'Background')  #
        # PALETTE = [[255, 255, 255], [255, 204, 0], [0, 0, 255], [0, 255, 128], [0, 204, 255],
        #            [0, 0, 0]]  # [255, 255, 255],
        mask = mask.astype(np.uint8)
        result = cv2.addWeighted(img, 1.0, mask.astype(np.uint8), 0.2, 0)
        #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        plt.show()
        #cv2.imshow(filename, result)
        #cv2.waitKey(0)




read_directory("E:/Datasets/SeaIce/seaice/train/image/")