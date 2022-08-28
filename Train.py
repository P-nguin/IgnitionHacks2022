from asyncio.windows_events import NULL
from cProfile import run
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def train(trainLoader, testLoader):
    device = NULL
    if(torch.cuda.is_available()): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 10), nn.LogSoftmax(dim=1))
    crit = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    epochs = 3
    steps = 0
    runningLoss = 0
    whenPrint = 10
    trainLosses = []
    testLosses = []

    for epoch in range(epochs):
        for inputs, labels in trainLoader:
            labels = labels.to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = crit(logps, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            
            if steps % whenPrint == 0:
                testLoss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testLoader:
                        labels = labels.to(device)
                        inputs = inputs.to(device)
                        logps = model.forward(inputs)
                        batch_loss = crit(logps, labels)
                        testLoss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        trainLosses.append(runningLoss/len(trainLoader))
                        testLosses.append(testLoss/len(testLoader))
                        runningLoss = 0
                        model.train()
    torch.save(model, 'PleaseWork.pth')