import torch
from  torch.utils.data import Dataset

class BengaliDataset(Dataset):

    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform
        self.labels = self.data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()


        a = self.data.iloc[idx,4:]
        x = a.to_numpy()
        x = x.astype('float').reshape(1,137,236)
        y = self.labels.iloc[idx]
        y = y.to_numpy()
        y = y.reshape(-1,3)
        sample = {'data':x, 'labels':y}
        if(self.transform):
            sample['data'] = self.transform(sample['data'])
        return sample
