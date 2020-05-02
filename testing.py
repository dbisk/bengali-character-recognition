import matplotlib.pyplot as plt
from utils import getData, trainModel
from torch.utils.data import DataLoader
from torchvision import transforms

def testDataset():
    train, test = getData('./dataset/', 'train.csv')
    print('Testing...')
    print(train.data.iloc[0])
    print(train[0])
    print(train[[0,1]]['data'].shape)
    print(train[[0,1]]['labels'])
    print(train[[0,1]]['identifier'])
    for i in range(5):
        plt.subplot(150 + i + 1)
        sample = train[i]['data'][0][0]
        plt.imshow(sample.astype(int),cmap='gray', vmin=0, vmax=255)
    plt.show()



def testTrainingLoop():

    main_transform = transforms.Compose([transforms.ToTensor()])
    trainset, testset = getData('./dataset/', 'train.csv', train_transform = main_transform, test_transform = main_transform)
    print(trainset[0])
    train_dataloader = DataLoader(trainset, batch_size = 16, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(testset, batch_size = 16, shuffle = True, num_workers = 8)
    trainModel(None, train_dataloader, test_dataloader)


#testDataset()
testTrainingLoop()
