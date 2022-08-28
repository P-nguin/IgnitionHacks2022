import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

dataDir = '/data/train'
def loadTrainingData(dataDir, validSize = .2):
    trainTransforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
    testTransforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),])
    trainData = datasets.ImageFolder(dataDir, transform = trainTransforms)
    testData = datasets.ImageFolder(dataDir,transform = testTransforms)
    numTrain = len(trainData)
    indices = list(range(numTrain))
    split = int(np.floor(validSize * numTrain))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    trainIdx = indices[split:]
    testIdx = indices[:split]
    trainSampler = SubsetRandomSampler(trainIdx)
    testSampler = SubsetRandomSampler(testIdx)
    trainLoader = torch.utils.data.DataLoader(trainData, sampler = trainSampler, batchSize=64)
    testLoader = torch.utils.data.DataLoader(testData, sampler = testSampler, batchSize=64)
    return trainLoader, testLoader
trainLoader, testLoader = loadTrainingData(dataDir, .2)
print(trainLoader.dataset.classes)