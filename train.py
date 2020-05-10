# import 3rd-party modules
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def makeSquareBatch(img_batch, size):
    batch_size, channels, height, width = img_batch.shape
    squared_images = torch.ones((batch_size, channels, size, size), dtype=img_batch.dtype) * torch.max(img_batch)

    squared_images[:, :, (size-height)//2:(size-height)//2 + height, (size-width)//2:(size-width)//2 + width] = img_batch
    return squared_images

def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using:", device)

    # send model to device
    model = model.to(device)

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # save the best model
    best_acc = 0.0
    # begin training loop
    for epoch in range(epochs):
        model.train() # set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            inputs = batch['data']
            inputs = makeSquareBatch(inputs, 236).to(device)
            labels = batch['labels'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out_root, out_vowe, out_cons = model(inputs)
            loss_root = criterion(out_root, labels[:,0])
            loss_vowe = criterion(out_vowe, labels[:,1])
            loss_cons = criterion(out_cons, labels[:,2])
            loss = loss_root + loss_vowe + loss_cons
            loss.backward()
            optimizer.step()

            # check whether our prediction was correct
            preds_root = torch.max(out_root, 1)[1]
            preds_vowe = torch.max(out_vowe, 1)[1]
            preds_cons = torch.max(out_cons, 1)[1]
            correct += torch.stack(((preds_root == labels[:,0]), (preds_vowe == labels[:,1]), (preds_cons == labels[:,2])), dim=1).all(1).sum().item()
            # correct += (preds_vowe == labels[:,1]).sum().item()
            # correct += (preds_cons == labels[:,2]).sum().item()
            total += labels.size(0)

            # statistics
            running_loss += loss.item()
        
        # normalize by number of batches
        running_loss /= len(train_dataloader)
        acc = 100 * correct / total
        print("[%d] loss %.3f, acc %.3f" % (epoch + 1, running_loss, acc))
        if (acc > best_acc):
            torch.save(model, './best_model.pth')
            best_acc = acc

        # every few epochs, check validation set
        if (epoch % 5 == 0 or epoch + 1 == epochs ):
            model.eval() # set model to eval mode
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
                    inputs = batch['data']
                    inputs = makeSquareBatch(inputs, 236).to(device)
                    labels = batch['labels'].to(device)
                    out_root, out_vowe, out_cons = model(inputs)
                    preds_root = torch.max(out_root, 1)[1]
                    preds_vowe = torch.max(out_vowe, 1)[1]
                    preds_cons = torch.max(out_cons, 1)[1]
                    correct += torch.stack(((preds_root == labels[:,0]), (preds_vowe == labels[:,1]), (preds_cons == labels[:,2])), dim=1).all(1).sum().item()
                    # correct += (preds_root == labels[:,0]).sum().item()
                    # correct += (preds_vowe == labels[:,1]).sum().item()
                    # correct += (preds_cons == labels[:,2]).sum().item()
                    total += labels.size(0)
            print("[%d] VAL acc %.3f" % (epoch + 1, 100 * correct / total))

        # step the lr scheduler
        scheduler.step()
    
    # complete, return the model
    return model
