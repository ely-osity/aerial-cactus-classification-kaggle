import pandas as pd
import glob
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os

from model import cnn

LR = 1e-2
epoches = 10

class dtset(Dataset):
    def __init__(self, root_dir, img_path_list, label_list):
        self.root_dir = root_dir
        self.label_list = label_list
        self.img_path_list = img_path_list   
    
    def __getitem__(self, index):                #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img_name = self.img_path_list[index]
        label = self.label_list[index]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        img=transforms.ToTensor()(img)

        return img, label
# 返回长度
    def __len__(self):
        return len(self.img_path_list)
    


train_list = pd.read_csv("D:\\jpytr\DL\\cactus\\aerial-cactus-identification\\train.csv", dtype={'id': str})

root_dir = "D:\\jpytr\DL\\cactus\\aerial-cactus-identification\\train\\train\\"
child_dir = train_list['id']
label_list = train_list['has_cactus']

cactus_dataset = dtset(root_dir, child_dir, label_list)

smaller_size = int(0*len(cactus_dataset))
bigger_size = len(cactus_dataset) - smaller_size
smaller_set, bigger_set = data.random_split(cactus_dataset, [smaller_size, bigger_size])
train_loader = DataLoader(bigger_set, batch_size=32, shuffle=False)
#test_loader = DataLoader(smaller_set, batch_size=32, shuffle=True)

model = cnn(3, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

#train
for epoch in range(epoches):
    print('epoch{%d}:'%(epoch+1))
    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = Variable(img)
        data = Variable(label)

        #forward
        out=model(img)
        loss=criterion(out, label)
        #torch upgraded version use item()
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_corect = (pred==label).sum()
        accuracy = (pred==label).float().mean()
        running_acc += num_corect.item()

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss:{:.6f}, Acc:{:.6f}'.format(running_loss/(len(bigger_set)), running_acc/(len(bigger_set))))
    
    #test
    # model.eval()
    # eval_loss = 0
    # eval_acc = 0
    # for data in test_loader:
    #     img, label = data
    #     img = Variable(img, volatile=True)
    #     data = Variable(label, volatile=True)

    #     out = model(img)
    #     loss = criterion(out, label)
    #     eval_loss += loss.item() * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     num_correct = (pred == label).sum()
    #     eval_acc += num_correct.item()
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     smaller_set)), eval_acc / (len(smaller_set))))
    # print()

torch.save(model.state_dict(), 'model.pth')