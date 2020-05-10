
from tqdm import tqdm
import torch
import torch.nn as nn
from models import customModel1 as custom
from models import customModel2 as custom2

def train(model, train_dataloader, test_dataloader, pretrained = False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)
    lr = .01
    model = custom2.CustomNet()
    #if(pretrained):
    #    model.load_state_dict(torch.load('./models/trainedModels/currBest.model'))
    #    print('Pretrained custom model loaded')
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=.1)
    epochs = 51
    currBestAcc = 0
    bestEpoch = 0
    for epoch in range(epochs):

        model.train()
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
            outs1 = model(imgs, 1)
            outs2 = nn.functional.pad(model(imgs,2), 157)
            outs3 = nn.functional.pad(model(imgs,3), 161)
            outs = torch.cat((outs1,outs2,outs3),1)
            _,preds1 = torch.max(outs1,1)
            _,preds2 = torch.max(outs2,1)
            _,preds3 = torch.max(outs3,1)
            preds = torch.cat((preds1, preds2, preds3),1)
            #from pytorch tutorials:
            correct += (preds == labels).all(1).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            l = loss(outs, labels)
            totalLoss += l.item()
            l.backward()
            optimizer.step()
        scheduler.step()
        print('Loss:', totalLoss/cnt)
        print('Accuracy:', correct / total)
        if(epoch % 10 == 0):
            model.eval()
            print('Testing on Validation')
            correct = 0
            total = 0
            for i, batch in enumerate(tqdm(test_dataloader)):
                imgs = batch['data']
                labels = batch['labels']
                imgs = imgs.float()
                labels = labels.squeeze(1)
                imgs = imgs.to(device)
                labels = labels.to(device)
                outs = model(imgs)
                _,preds = torch.max(outs,1)
                correct += (preds == labels).all(1).sum().item()
                total += labels.size(0)
            print('Testing accuracy:', correct/total)
            if(correct/total > currBestAcc):
                currBestAcc = correct/total
                torch.save(model.state_dict(), './models/trainedModels/multiNet.model')
                bestEpoch = epoch



    print('Epoch', bestEpoch, 'gave the best validation accuracy of', currBestAcc)

    return

