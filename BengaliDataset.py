import numpy as np
import torch
from  torch.utils.data import Dataset
import pyarrow.parquet as pq
import pandas as pd
#import torchvision
class BengaliDataset(Dataset):

    def __init__(self, csvfile, parquetDir, train=True, transform=None):
        if(train):
            print("Initializing training dataset...")
        else:
            print("Initializing testing dataset...")

        self.labels = pd.read_csv(parquetDir + csvfile)
        self.root_dir = parquetDir
        self.transform = transform
        base_str = 'train_image_data_'
        filetype = '.parquet'

        startpoint = 0 if train else 3

        self.data = pq.read_pandas(self.root_dir  + base_str + str(startpoint) + filetype).to_pandas()
        if(train):
            print("1 out of 3 frames finished")
        else:
            print("Done initializing")
        if(train):
            d1 = pq.read_pandas(self.root_dir  + base_str + str(startpoint + 1) + filetype).to_pandas()
            print("2 out of 3 frames finished")
            d2 = pq.read_pandas(self.root_dir  + base_str + str(startpoint + 2) + filetype).to_pandas()
            self.data = pd.concat([self.data, d1,d2], ignore_index=True)
            print("Done initializing")
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()


        a = self.data.iloc[idx,1:]
        x =a.to_numpy()
        x = x.astype('float').reshape(-1,1,137,236)
        y = self.labels.iloc[idx,1:4]
        y = y.to_numpy()
        y = y.reshape(-1,3)

        if(self.transform):
            x = self.transform(x)
        return {'data':x, 'labels':y}

