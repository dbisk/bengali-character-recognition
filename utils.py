import pyarrow.parquet as pq
import pandas as pd
from BengaliDataset import BengaliDataset as bd
import train
import numpy as np


def getData(root_dir, csvfile, train_transform=None, test_transform=None):

    base_str = 'train_image_data_'
    filetype = '.parquet'
    labels = pd.read_csv(root_dir + csvfile)
    data1 = pq.read_pandas(root_dir + base_str + '0' + filetype).to_pandas()
    print('First parquet read')
    data2 = pq.read_pandas(root_dir + base_str + '1' + filetype).to_pandas()
    print('Second parquet read')
    data3 = pq.read_pandas(root_dir + base_str + '2' + filetype).to_pandas()
    print('Third parquet read')
    data4 = pq.read_pandas(root_dir + base_str + '3' + filetype).to_pandas()
    print('Fourth parquet read')
    data = pd.concat([data1, data2, data3, data4])
    print('All data loaded')
    data.insert(0, 'grapheme_root', labels['grapheme_root'])
    data.insert(1, 'vowel_diacritic', labels['vowel_diacritic'])
    data.insert(2, 'consonant_diacritic', labels['consonant_diacritic'])

    del(labels)
    del(data1, data2, data3, data4)
    data.astype(np.int8, errors='ignore', copy=False)
    train = bd(data.iloc[:160672], transform=train_transform)
    test = bd(data.iloc[160672:], transform=test_transform)
    return train, test


def trainModel(net, train_dataloader, test_dataloader):

    train.train(net, train_dataloader, test_dataloader)
        #forward

        #backward

        #validation
    pass
