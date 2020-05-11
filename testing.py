import matplotlib.pyplot as plt
from utils import getData, trainModel, dsetToPickle
from torch.utils.data import DataLoader
from torchvision import transforms
from models import customModel1 as custom
import torch

def testDataset(need_pickle=False):
    if (need_pickle):
        pkl_path = dsetToPickle('./dataset/', 'train.csv')
    else:
        pkl_path = './dataset/full_data.pkl'
    train, test = getData(pkl_path)
    print('Testing...')
    print("train.data:\n", train.data)
    print("train[0]:\n", train[0])
    for i in range(5):
        plt.subplot(150 + i + 1)
        sample = train[i]['data'][0]
        plt.imshow(sample.astype(int), cmap='gray', vmin=0, vmax=255)
    plt.show()



def testTrainingLoop(need_pickle=False):
    if (need_pickle):
        pkl_path = dsetToPickle('./dataset/', 'train.csv')
    else:
        pkl_path = './dataset/full_data.pkl'

    main_transform = transforms.Compose([transforms.ToTensor()])
    trainset, testset = getData(pkl_path, train_transform = main_transform, test_transform = main_transform)
    train_dataloader = DataLoader(trainset, batch_size = 64, shuffle = True)
    test_dataloader = DataLoader(testset, batch_size = 64, shuffle = True)
    trainModel(None, train_dataloader, test_dataloader)

def visualize():

    pkl_path = './dataset/full_data.pkl'
    main_transform = transforms.Compose([transforms.ToTensor()])
    trainset, testset = getData(pkl_path, train_transform = main_transform, test_transform = main_transform)
    test_dataloader = DataLoader(testset, batch_size = 1, shuffle = True)
    model = custom.CustomNet()
    plt.figure(dpi=300)
    for i,batch in enumerate(test_dataloader):
        if(i <5):
            plt.subplot(150 + i + 1)
            data = test_dataloader[i]
            img = data['data']
            labels = data['labels']
            outs = model(img)
            _,preds = torch.max(outs,1)
            print(labels, preds)
        else:
            break

if __name__ == '__main__':
    # set this to true if you don't already have a pickle file
    #testDataset(need_pickle=False)
    #testTrainingLoop(need_pickle=False)
    visualize()
