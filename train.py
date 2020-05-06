
from tqdm import tqdm
import torch
import torch.nn as nn
from models import customModel1 as custom

def train(model, train_dataloader, test_dataloader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)
    lr = .005
    model = custom.CustomNet()
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    epochs = 10
    for epoch in range(epochs):
        totalLoss = 0
        cnt = 0
        correct = 0
        total = 0
        print('Epoch:', epoch)
        for i, batch in enumerate(tqdm(train_dataloader)):
            cnt += 1
            imgs = batch['data']
            labels = batch['labels']
            imgs = imgs.float()
            labels = labels.squeeze(1)
            imgs = imgs.to(device)
            labels = labels.to(device)
            outs = model(imgs)

            #from pytorch tutorials:
            _,preds = torch.max(outs,1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            l = loss(outs, labels)
            totalLoss += l.item()
            l.backward()
            optimizer.step()
        print('Loss:', totalLoss/cnt)
        print('Accuracy:', correct / total)

    return

