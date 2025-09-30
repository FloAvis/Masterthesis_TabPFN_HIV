import torch
import torch.utils.data as data
import numpy as np
import scipy.io as scio
import os
from sklearn.model_selection import KFold
import math

class DatasetLoader(data.Dataset):
    def __init__(self, X, y, data_inds=None, batch_size=128, shuffle=False):
        super(DatasetLoader, self).__init__()
        self.X = X
        self.y = y
        self.data_inds = data_inds
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            self.index = torch.randperm(self._len())
        else:
            self.index = torch.arange(0, self._len(), dtype=torch.int)
        self.start = 0
        return self
    
    def __next__(self):
        if self.start < self._len():
            self.end = min(self.start+self.batch_size, self._len())
            X = self.X[self.data_inds[self.index[self.start:self.end]]]
            y = self.y[self.data_inds[self.index[self.start:self.end]]]
            self.start = self.end
            if X.dim() == 1:
                X = X.unsqueeze(0)
                y = y.unsqueeze(0)
            return X, y
        else:
            raise StopIteration
        
    def _len(self):
        return len(self.data_inds)
    
    def __len__(self):
        return math.ceil(self._len() / self.batch_size)

class Dataset():
    def __init__(self, datadir = "./Datasets", configs={},
                 nfold=10):
        self.datadir = datadir
        self.datafile = os.path.join(datadir, self.name(), self.name()+'.mat')
        
        # Load data
        self.dtype = configs['dtype']
        self.data_standardizing = configs['data_standardizing']
        self.eps = configs['eps']
        self._data(nfold, configs['shuffle'], configs['rand_seed'])
        self.feat_dim = self.X.size(1)
        self.num_class = self.y.size(1)

        # Create iterable data loaders
        self.train_dataloader = DatasetLoader(self.X, self.y,
                                              batch_size=configs['train_batch_size'],
                                              shuffle=configs['shuffle'])
        self.test_dataloader = DatasetLoader(self.X, self.y,
                                             batch_size=configs['test_batch_size'],
                                             shuffle=False)
        self.val_dataloader = DatasetLoader(self.X_val, self.y_val,
                                            data_inds=np.arange(self.X_val.size(0)),
                                            batch_size=configs['test_batch_size'],
                                            shuffle=False)
        
    def name(self):
        return self.__class__.__name__
    
    def size(self):
        return self.train_dataloader._len()
        
    def data_cv_splitter(self, fold, nfold=10, shuffle=False, random_state=None):
        train_inds_file = self._get_path(fold, 'train', nfold, shuffle, random_state)
        test_inds_file = self._get_path(fold, 'test', nfold, shuffle, random_state)
        if not os.path.exists(train_inds_file):
            self.dataset_split(nfold, shuffle, random_state)
        data = scio.loadmat(train_inds_file)
        self.train_dataloader.data_inds = data['index'][0]
        data = scio.loadmat(test_inds_file)
        self.test_dataloader.data_inds = data['index'][0]
            
    def dataset_split(self, nfold=10, shuffle=False, random_state=None):
        '''
        Split dataset for n-fold cross-validation.
    
        Parameters
        ----------
        nfold : int, optional
            Number of fold for cross-validation. The default is 5.
        shuffle : bool, optional
            If shuffle=True, shuffling the data before cross-validation. The default is False.
        random_state : int, optional
            When shuffle is True, random_state affects the ordering of the indices,
            which controls the randomness of each fold. Otherwise, this parameter
            has no effect. The default is None.
        '''
        print('Split {:s} for {:d}-fold cross-validation.'.format(self.name(), nfold))
        
        # Creating Dataset spliter
        spliter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)
        
        # Splitting dataset
        count = 1
        for train_inds, test_inds in spliter.split(self.X):
            train_inds_file = self._get_path(count, 'train', nfold, shuffle, random_state)
            test_inds_file = self._get_path(count, 'test', nfold, shuffle, random_state)
            scio.savemat(train_inds_file, {'index': train_inds})
            scio.savemat(test_inds_file, {'index': test_inds})
            count += 1
        
    def _data(self, nfold=10, shuffle=False, random_state=None):
        self._data_loading()
        self._data_preprocess()
        self._val_data_split(nfold, shuffle, random_state)
        
    def _data_loading(self):
        data = scio.loadmat(self.datafile)
        X_train = torch.from_numpy(data['X_train'].astype(np.float)).type(self.dtype)
        X_test = torch.from_numpy(data['X_test'].astype(np.float)).type(self.dtype)
        Y_train = torch.from_numpy(data['Y_train']).type(self.dtype)
        Y_test = torch.from_numpy(data['Y_test']).type(self.dtype)
        self.X = torch.cat((X_train, X_test), dim=0)
        self.y = torch.cat((Y_train, Y_test), dim=0)
            
    def _data_preprocess(self):
        if self.data_standardizing:
            max_X = torch.max(self.X, dim=0, keepdim=True)[0]
            min_X = torch.min(self.X, dim=0, keepdim=True)[0]
            self.X = (self.X - min_X) / (max_X - min_X + self.eps)
    
    def _val_data_split(self, nfold=10, shuffle=False, random_state=None):
        split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        traintest_inds_file = os.path.join(self.datadir, self.name(), self.name()+split_config+'_traintest.mat')
        val_inds_file = os.path.join(self.datadir, self.name(), self.name()+split_config+'_val.mat')
        if not os.path.exists(traintest_inds_file):
            spliter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state)
            for traintest_inds, val_inds in spliter.split(self.X):
                scio.savemat(traintest_inds_file, {'index': traintest_inds})
                scio.savemat(val_inds_file, {'index': val_inds})
                break
        data = scio.loadmat(val_inds_file)
        self.X_val = self.X[data['index'][0]]
        self.y_val = self.y[data['index'][0]]
        data = scio.loadmat(traintest_inds_file)
        self.X = self.X[data['index'][0]]
        self.y = self.y[data['index'][0]]
    
    def _get_path(self, fold, data_type='train', nfold=10, shuffle=False, random_state=None):
        split_config = '_{:d}_{:d}_{:d}_'.format(shuffle, random_state, nfold)
        fileName = os.path.join(self.datadir, self.name(), self.name()+
                                split_config+'_{:s}_cv{:d}.mat'.format(data_type, fold))
        return fileName
             
class emotions(Dataset):
    pass
        
class genbase(Dataset):
    pass
    
class medical(Dataset):
    pass

class enron(Dataset):
    pass

class scene(Dataset):
    pass

class yeast(Dataset):
    pass

class corel5k(Dataset):
    pass

class rcv1subset1(Dataset):
    pass

class rcv1subset2(Dataset):
    pass

class bibtex(Dataset):
    pass

class delicious(Dataset):
    pass

class iaprtc12(Dataset):
    pass

class espgame(Dataset):
    pass

class mirflickr(Dataset):
    pass

class tmc2007(Dataset):
    pass

class mediamill(Dataset):
    pass

class Dataset2(Dataset):
    def _data_loading(self):
        data = scio.loadmat(self.datafile)
        self.X = torch.from_numpy(data['X'].astype(np.float)).type(self.dtype)
        self.y = torch.from_numpy(data['Y']).type(self.dtype)
    
class CAL500(Dataset2):
    pass

class language_log(Dataset2):
    pass

class Image(Dataset2):
    pass

class slashdot(Dataset2):
    pass

class eurlex_directory_codes(Dataset2):
    pass

class eurlex_subject_matters(Dataset2):
    pass

class bookmarks(Dataset2):
    pass

class Dataset3(Dataset):
    def _data_loading(self):
        data = scio.loadmat(self.datafile)
        X_train = torch.from_numpy(data['X_train'].astype(np.float)).type(self.dtype)
        X_test = torch.from_numpy(data['X_test'].astype(np.float)).type(self.dtype)
        X_test3 = torch.from_numpy(data['X_test3'].astype(np.float)).type(self.dtype)
        Y_train = torch.from_numpy(data['Y_train']).type(self.dtype)
        Y_test = torch.from_numpy(data['Y_test']).type(self.dtype)
        Y_test3 = torch.from_numpy(data['Y_test3']).type(self.dtype)
        self.X = torch.cat((X_train, X_test, X_test3), dim=0)
        self.y = torch.cat((Y_train, Y_test, Y_test3), dim=0)
    
class Corel16k001(Dataset3):
    pass

class Corel16k002(Dataset3):
    pass