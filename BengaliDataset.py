import pandas as pd
import torch
from torch.utils.data import Dataset

class BengaliDataset(Dataset):
    
    HEIGHT = 137
    WIDTH = 236

    def __init__(self, dataframe, transform=None):
        label_list = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        self.labels = dataframe[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].copy(deep=True)
        self.data = dataframe.copy(deep=False)
        self.data.drop(columns=label_list, inplace=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if (torch.is_tensor(idx)):
            idx = idx.tolist()
        
        img = self.data.iloc[idx,:].to_numpy().reshape((-1, self.HEIGHT, self.WIDTH))
        lbl = self.labels.iloc[idx,:].to_numpy()

        if (self.transform):
            img = self.transform(img)
        return {
            'data': img,
            'labels': lbl,
        }
