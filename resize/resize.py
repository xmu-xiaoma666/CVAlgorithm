import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from PIL import Image
import numpy as np
import torch
import PIL
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


imgpath='./data/person.jpg'
# 使用skimage读取图像
img_skimage = io.imread(imgpath)        # skimage.io imread()-----np.ndarray,  (H x W x C), [0, 255],RGB
print(img_skimage.shape)
print(type(img_skimage))

# 使用opencv读取图像
img_cv = cv2.imread(imgpath)            # cv2.imread()------np.array, (H x W xC), [0, 255], BGR
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)      # BGR->RGB
print(img_cv.shape)
print(type(img_cv))

# 使用PIL读取
img_pil = Image.open(imgpath)         # PIL.Image.Image对象
img_pil_1 = np.array(img_pil)           # (H x W x C), [0, 255], RGB
print(img_pil_1.shape)
print(type(img_pil_1))


### Show
# plt.figure()
# for i, im in enumerate([img_skimage, img_cv, img_pil_1]):
#     ax = plt.subplot(1, 3, i + 1)
#     ax.imshow(im)
# plt.show()

# ------------np.ndarray转为torch.Tensor------------------------------------
# numpy image: H x W x C
# torch image: C x H x W
tensor_skimage = torch.from_numpy(np.transpose(img_skimage, (2, 0, 1)))
tensor_cv = torch.from_numpy(np.transpose(img_cv, (2, 0, 1)))
tensor_pil = torch.from_numpy(np.transpose(img_pil_1, (2, 0, 1)))



resize=Compose([
        Resize((112,112),PIL.Image.BICUBIC),
        # ToTensor()
    ])

plt.figure()
plt.imshow(resize(img_pil))
plt.show()
# print(resize(img_pil).shape)