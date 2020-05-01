import matplotlib.pyplot as plt
from BengaliDataset import BengaliDataset
from utils import getData

#imgs = pq.read_pandas('./dataset/train_image_data_1.parquet').to_pandas()
#
#test = imgs.iloc[0]
#test = np.array(test)
#test = test[1:]
#test = test.reshape((137,236)).astype(int)
#
#plt.figure()
#plt.imshow(test,cmap='gray', vmin=0, vmax=255)
#plt.show()

train, test = getData('./dataset/', 'train.csv', write = True)
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
#sample = d[5]['data'][0][0]
#plt.figure()
#plt.imshow(sample.astype(int), cmap='gray', vmin=0, vmax=255)
#plt.show()
