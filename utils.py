# import built-in modules
import time
from multiprocessing import Pool

# import 3rd-party modules
import pandas as pd

# import custom modules
from BengaliDataset import BengaliDataset
import train

# def test_fn(filepath):
#     return pd.read_parquet(filepath, engine='pyarrow')

def dsetToPickle(root_dir, csvfile):
    """Converts the parquet files to pickle format for faster io"""
    base_str = 'train_image_data_'
    labels = pd.read_csv(root_dir + csvfile).set_index('image_id', drop=True)
    filepaths = []
    for i in range(4):
        filepaths.append(root_dir + base_str + str(i) + '.parquet')
    start_time = time.time()
    print("Reading parquet files...")
    # with Pool(processes=4) as pool:
    #     data = pool.map(test_fn, filepaths)
    # print(data)
    data = []
    for i in range(4):
        data.append(pd.read_parquet(filepaths[i], engine='pyarrow').set_index('image_id', drop=True))
        print("Loaded parquet file " + str(i) + "...")
    data = pd.concat(data, copy=False)
    print("Parquet loading completed. Elapsed: %d seconds" % (time.time() - start_time))

    # insert labels to the front of the dataframe
    labels = labels.iloc[:len(data)]
    data.insert(0, 'grapheme_root', labels['grapheme_root'])
    data.insert(1, 'vowel_diacritic', labels['vowel_diacritic'])
    data.insert(2, 'consonant_diacritic', labels['consonant_diacritic'])
    del(labels) # probably not required, but clears up some memory

    # shuffle
    data = data.sample(frac=1)

    # save to pickle file
    pkl_path = root_dir + "full_data.pkl"
    data.to_pickle(pkl_path)
    del(data)
    return pkl_path

def getData(pickle_path, split=0.8, drop=0, train_transform=None, test_transform=None):
    start_time = time.time()
    print("Reading pickle from " + pickle_path + "...")
    data = pd.read_pickle(pickle_path)
    print("Pickle loading completed. Elapsed: %d seconds" % (time.time() - start_time))
    if (drop > 0 and drop < 1):
        data.drop(data.tail(int(len(data) * drop)).index, inplace=True)
    split_index = int(split * len(data))

    # create the BengaliDataset objects
    train = BengaliDataset(data.iloc[:split_index], transform=train_transform)
    test = BengaliDataset(data.iloc[split_index:], transform=test_transform)
    return train, test


def trainModel(net, train_dataloader, test_dataloader):

    train.train(net, train_dataloader, test_dataloader)
        #forward

        #backward

        #validation
    pass
