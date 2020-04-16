import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from BengaliDataset import BengaliDataset

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

d = BengaliDataset('train.csv', './dataset/')
print(d.data.shape)
print(d[[0,1]]['data'].shape)
print(d[[0,1]]['labels'])
sample = d[5]['data'][0][0]
plt.figure()
plt.imshow(sample.astype(int), cmap='gray', vmin=0, vmax=255)
plt.show()
