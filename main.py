# import built-in modules
import time

# import 3rd-party modules
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# import custom modules
import train
import utils
from models import prnet

NEED_PICKLE = False
TOTAL_ROOTS = 168
TOTAL_VOWELS = 11
TOTAL_CONS = 8

def main():
    if (NEED_PICKLE):
        pkl_path = utils.dsetToPickle('./dataset/', 'train.csv')
    else:
        pkl_path = './dataset/full_data.pkl'
    
    # set up the data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45], std=[0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45], std=[0.225]),
    ])

    # set up the datasets/loaders
    trainset, testset = utils.getData(pkl_path, split=0.75, drop=0.5, train_transform=train_transform, test_transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True, drop_last=True)
    
    # create the model
    model = prnet.PretrainedResnet(TOTAL_ROOTS, TOTAL_VOWELS, TOTAL_CONS)
    # model = torch.load('./saved_model_best.pth')

    # train the model
    model = train.train(model, trainloader, testloader, epochs=35, lr=0.01)

    # save the model
    torch.save(model, './saved_model.pth')

    # validate the model
    # acc = train.validate(model, testloader)
    # print("Validation Accuracy: %.3f" % (acc))

    return

if __name__ == '__main__':
    main()
