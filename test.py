import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy
import os

from model import cnn

class test_dtset(Dataset):
    def __init__(self, root_dir, img_path_list):
        self.root_dir = root_dir
        self.img_path_list = img_path_list   
    
    def __getitem__(self, index):                #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img_name = self.img_path_list[index]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        img=transforms.ToTensor()(img)

        return img
# 返回长度
    def __len__(self):
        return len(self.img_path_list)
    

root_dir = "D:\\jpytr\DL\\cactus\\aerial-cactus-identification\\test\\test\\"
test_list = pd.read_csv("D:\\jpytr\\DL\\cactus\\aerial-cactus-identification\\sample_submission.csv", dtype={'id': str})

test_data = test_dtset(root_dir, test_list['id'])
test_loader = DataLoader(test_data, batch_size=32, shuffle = False)

model = cnn(3,2)
model.load_state_dict(torch.load('D:\\jpytr\DL\\cactus\\model.pth'))

model.eval()

outcomes = []

for data in test_loader:
    img = data
    img = Variable(img)
    outputs = model(img)
    predicted, index  = torch.max(outputs, 1)#??????????????
    degre = int(index[0])

    outcomes.append(degre)

test_outs = pd.DataFrame(outcomes)

test_list['has_cactus'] = test_outs
print(test_list[:10])

test_list.to_csv("D:\\jpytr\DL\\cactus\\result.csv", index=False)
print('*****************over******************')