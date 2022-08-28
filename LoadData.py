import numpy as np
import torch
from torchvision import datasets, transforms

dataDir = 'dataset/GarbageClassification'
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
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size = 64, sampler = trainSampler)
    testLoader = torch.utils.data.DataLoader(testData, batch_size = 64, sampler = testSampler)
    return trainLoader, testLoader
trainLoader, testLoader = loadTrainingData(dataDir, .2)