# import 3rd-party modules
from tqdm import tqdm
import torch
import torch.nn as nn

# FROM SUHAAS BRANCH FOR MULTINET
#from models import customModel1 as custom
#from models import customModel2 as custom2
#
#def train(model, train_dataloader, test_dataloader, pretrained = False):
#
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    print('Using:', device)
#    lr = .1
#    model = custom2.CustomNet()
#    #if(pretrained):
#    #    model.load_state_dict(torch.load('./models/trainedModels/currBest.model'))
#    #    print('Pretrained custom model loaded')
#    model = model.to(device)
#    loss = nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=.1)
#    epochs = 51
#    currBestAcc = 0
#    bestEpoch = 0
#    for epoch in range(epochs):
#
#        model.train()
#        totalLoss = 0
#        cnt = 0
#        correct = 0
#        total = 0
#        print('Epoch:', epoch)
#        for i, batch in enumerate(tqdm(train_dataloader)):
#            cnt += 1
#            imgs = batch['data']
#            labels = batch['labels']
#            imgs = imgs.float()
#            labels = labels.squeeze(1)
#            imgs = imgs.to(device)
#            labels = labels.to(device)
#            outs1 = model(imgs, 1)
#            outs2 = nn.functional.pad(model(imgs,2), (0,157))
#            outs3 = nn.functional.pad(model(imgs,3), (0,161))
#            outs = torch.stack((outs1,outs2,outs3), 2)
#            preds1 = torch.max(outs1,1)[1].view(-1,1)
#            preds2 = torch.max(outs2,1)[1].view(-1,1)
#            preds3 = torch.max(outs3,1)[1].view(-1,1)
#            preds = torch.cat((preds1, preds2, preds3),1)
#            #from pytorch tutorials:
#            correct += (preds == labels).all(1).sum().item()
#            total += labels.size(0)
import torch.optim as optim

def makeSquareBatch(img_batch, size):
    batch_size, channels, height, width = img_batch.shape
    squared_images = torch.ones((batch_size, channels, size, size), dtype=img_batch.dtype) * torch.max(img_batch)

    squared_images[:, :, (size-height)//2:(size-height)//2 + height, (size-width)//2:(size-width)//2 + width] = img_batch
    return squared_images

def validate(model, test_dataloader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
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
            total += labels.size(0)
    acc = 100 * correct / total
    return acc

def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using:", device)

    # send model to device
    model = model.to(device)

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ########################
    ## FROM SUHAAS BRANCH ##
    ########################
    # optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=.1)
    ########################

    # save the best model
    best_acc = 0.0
    # begin training loop
    for epoch in range(epochs):
        ##############################
        # SECTION FROM SUHAAS BRANCH #
        ##############################
        # totalLoss = 0
        # cnt = 0
        # correct = 0
        # total = 0
        # print('Epoch:', epoch)
        # for i, batch in enumerate(tqdm(train_dataloader)):
        #     cnt += 1
        #     imgs = batch['data']
        #     labels = batch['labels']
        #     imgs = imgs.float()
        #     labels = labels.squeeze(1)
        #     imgs = imgs.to(device)
        #     labels = labels.to(device)
        #     outs = model(imgs)

        #     #from pytorch tutorials:
        #     _,preds = torch.max(outs,1)
        #     correct += (preds == labels).sum().item()
        #     total += labels.size(0)
        ##############################
        # END SECTION  SUHAAS BRANCH #
        ##############################

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

# FROM SUHAAS MULTINET SECTION
#        scheduler.step()
#        print('Loss:', totalLoss/cnt)
#        print('Accuracy:', correct / total)
#        if(epoch % 10 == 0):
#            model.eval()
#            print('Testing on Validation')
#            correct = 0
#            total = 0
#            for i, batch in enumerate(tqdm(test_dataloader)):
#
#                imgs = batch['data']
#                labels = batch['labels']
#                imgs = imgs.float()
#                labels = labels.squeeze(1)
#                imgs = imgs.to(device)
#                labels = labels.to(device)
#                outs1 = model(imgs, 1)
#                outs2 = nn.functional.pad(model(imgs,2), (0,157))
#                outs3 = nn.functional.pad(model(imgs,3), (0,161))
#                outs = torch.stack((outs1,outs2,outs3), 2)
#                preds1 = torch.max(outs1,1)[1].view(-1,1)
#                preds2 = torch.max(outs2,1)[1].view(-1,1)
#                preds3 = torch.max(outs3,1)[1].view(-1,1)
#                preds = torch.cat((preds1, preds2, preds3),1)
#                #from pytorch tutorials:
#                correct += (preds == labels).all(1).sum().item()
#                total += labels.size(0)
#                #imgs = batch['data']
#                #labels = batch['labels']
#                #imgs = imgs.float()
#                #labels = labels.squeeze(1)
#               #imgs = imgs.to(device)
#               #labels = labels.to(device)
#                #outs = model(imgs)
#                #_,preds = torch.max(outs,1)
#                #correct += (preds == labels).all(1).sum().item()
#                #total += labels.size(0)
#            print('Testing accuracy:', correct/total)
#            if(correct/total > currBestAcc):
#                currBestAcc = correct/total
#                torch.save(model.state_dict(), './models/trainedModels/multiNet.model')
#                bestEpoch = epoch
#
#
#
#    print('Epoch', bestEpoch, 'gave the best validation accuracy of', currBestAcc)
        # every few epochs, check validation set
        if (epoch % 5 == 0 or epoch + 1 == epochs ):

            model.eval() # set model to eval mode
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
                    #########################
                    # FROM SUHAAS SECTION ###
                    #########################
                    # cnt += 1
                    # imgs = batch['data']
                    # labels = batch['labels']
                    # imgs = imgs.float()
                    # labels = labels.squeeze(1)
                    # imgs = imgs.to(device)
                    # labels = labels.to(device)
                    # outs = model(imgs)

                    # #from pytorch tutorials:
                    # _,preds = torch.max(outs,1)
                    # correct += (preds == labels).all(1).sum().item()
                    # total += labels.size(0)
                    ##########################
                    ##### END SECTION ########
                    ##########################
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
            acc = 100 * correct / total
            if (acc > best_acc):
                torch.save(model.state_dict(), './best_model.model')
                best_acc = acc
            print("[%d] VAL acc %.3f" % (epoch + 1, acc))

        # step the lr scheduler
        scheduler.step()

    # complete, return the model
    return model
