import threading
from tkinter import Variable
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

import LoadData
import Train

def check():
    threading.Timer(1.0, check())
    if len(os.listdir(dataDir)) > amt:
        amt = len(os.listdir(dataDir))
        
        data = datasets.ImageFolder(dataDir, transform=testTransforms)
        classes = data.classes
        idx = amt-1
        loader = torch.utils.data.DataLoader(data, batch_size=1)
        image, label = iter(loader).next()
        image = transforms.ToPILImage()(image)
        index = analyzeImg(image)

def get_random_images(num):
    data = datasets.ImageFolder(dataDir, transform=testTransforms)
    classes = data.classes
    ind = list(range(len(data)))
    np.random.shuffle(ind)
    idx = ind[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


def analyzeImg(img):
    imgTensor = testTransforms(img).float()
    imgTensor = imgTensor.unsqueeze_(0)
    input = Variable(imgTensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().np().argmax()
    return index

Train.train(LoadData.trainLoader, LoadData.testLoader)

testTransforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
device = Train.device
model = torch.load('PleaseWork.pth')
model.eval()

dataDir = 'imagesTaken'
amt = 0

#check()

to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
fig=plt.figure(figsize=(10,10))
classes = LoadData.trainLoader.dataset.classes
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = analyzeImg(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()