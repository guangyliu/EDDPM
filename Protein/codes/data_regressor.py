
"""
for loading data into model-ready dataloaders
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import  DataLoader
import pytorch_lightning as pl

from codes.utils import data_utils
from codes.nn.models import RelsoDiffusion

# ---------------------
# CONSTANTS
# ---------------------

ROOT_DATA_DIR = './data/'
# ENS_GRAD_TRAIN = ENS_GRAD_DIR + 'train_data.csv'
# ENS_GRAD_TEST = ENS_GRAD_DIR + 'test_data.csv'

MUT_GRAD_DIR = './data/mut_data/'
# ├── AMIE_PSEAE_test_data.csv
# ├── AMIE_PSEAE_train_data.csv
# ├── DLG_RAT_test_data.csv
# ├── DLG_RAT_train_data.csv
# ├── GB1_WU_test_data.csv
# ├── GB1_WU_train_data.csv
# ├── RASH_HUMAN_test_data.csv
# ├── RASH_HUMAN_train_data.csv
# ├── RL401_YEAST_test_data.csv
# ├── RL401_YEAST_train_data.csv
# ├── UBE4B_MOUSE_test_data.csv
# ├── UBE4B_MOUSE_train_data.csv
# ├── YAP1_HUMAN_test_data.csv
# └── YAP1_HUMAN_train_data.csv

MUT_DATASETS = ['GB1_WU',
                'GFP']


MUT_SEQUENCES = {
    'GFP': 'SKGEELFTGVVPILVELDGDVNGHKFNVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFESAMPEGHVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDYKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQDTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLESVTAAGITHGMDELYK',
    'Gifford': 'JJJJAAAAYDYWFDYJJJJJ',
    'GB1_WU': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'}


SEQ_DISTANCES_DIR = './data/seq_distances/'

# -------------------
# DATA MODULES
# -------------------
class EnsGradData(pl.LightningDataModule):
    """
    Gifford dataset
    """
    def __init__(self, data_dir=ROOT_DATA_DIR,
                        dataset=None,
                        task='recon',
                        train_val_split=0.7,
                        batch_size=100,
                        seqdist_cutoff=None,
                        encoder=None):
        super().__init__()

        if encoder is None:
            print("Encoder not provided")
        else:
            self.encoder = encoder.eval().cpu()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading data from: {self.data_dir}')

        # print(f'setting up seq distances')
        # self._setup_distances()
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()


    def _prepare_data(self):

        # load files
        train_df = pd.read_csv(str(self.data_dir + 'gifford_data/train_data.csv'))
        test_df = pd.read_csv(str(self.data_dir + 'gifford_data/test_data.csv'))

        train_seqs, train_fitness = data_utils.load_raw_giff_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_giff_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')


    def _setup_task(self):

        if self.task == 'next-step':

            # train set
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)

            # test set
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness]

        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit = train_test_split(*train_all_data_numpy,
                                                                    train_size=train_size,
                                                                    random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self, shuffle_bool=True):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def valid_dataloader(self, shuffle_bool=False):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def test_dataloader(self, shuffle_bool=False):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = shuffle_bool)


# ----------------------------
# Mutational classes
# ----------------------------

class MutData(pl.LightningDataModule):
    def __init__(self, data_dir=ROOT_DATA_DIR,
                                dataset='AMIE_PSEAE',
                                task='recon',
                                train_val_split=0.7,
                                batch_size=100,
                                seqdist_cutoff=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        self.train_data_file = '{}_train_data.csv'.format(dataset)
        self.test_data_file = '{}_test_data.csv'.format(dataset)

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()


    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.train_data_file), header=None)
        test_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.test_data_file),  header=None)


        train_seqs, train_fitness = data_utils.load_raw_mut_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_mut_data(test_df)

        # log scaling fitness
        # train_fitness += train_fitness.min()
        # test_fitness += test_fitness.min()

        # train_fitness, test_fitness = torch.log(train_fitness), torch.log(test_fitness)

        #train_fitness = train_fitness + torch.abs(train_fitness.min()) + 0.001
        #test_fitness = test_fitness + torch.abs(test_fitness.min()) + 0.001

        #train_fitness = torch.log(train_fitness)
        #test_fitness = torch.log(test_fitness)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        if self.task == 'next-step':
            # LSTM training dataset (next-char prediction)
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')


        #train_all_data = [train_data, train_targets, train_fitness, self.train_seq_dist]
        train_all_data = [train_data, train_targets, train_fitness]
        train_all_data_numpy = [x.numpy() for x in train_all_data ]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit = train_test_split(*train_all_data_numpy,
                                                                    train_size=train_size,
                                                                    random_state=42)


        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit]]

    
        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list)
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self, shuffle_bool=True):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def valid_dataloader(self, shuffle_bool=False):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def test_dataloader(self, shuffle_bool=False):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = shuffle_bool)



# ----------------------------
# TAPE data class
# ----------------------------

class TAPE(pl.LightningDataModule):
    def __init__(self, data_dir=ROOT_DATA_DIR,
                                dataset='TAPE',
                                task='next-step',
                                batch_size=100,
                                seqdist_cutoff=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        self.train_data_file = '{}_train_data.csv'.format(dataset)
        self.test_data_file = '{}_test_data.csv'.format(dataset)
        self.valid_data_file = '{}_valid_data.csv'.format(dataset)

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.test_data_file))
        valid_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.valid_data_file))


        train_seqs, train_fitness = data_utils.load_raw_tape_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_tape_data(test_df)
        valid_seqs, valid_fitness = data_utils.load_raw_tape_data(valid_df)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)
        self.raw_valid_tup = (valid_seqs, valid_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]
        self.valid_N = valid_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test/valid sizes: {(self.train_N, self.test_N, self.valid_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_task(self):

        if self.task == 'next-step':
            # LSTM training dataset (next-char prediction)
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            valid_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_valid_tup[0]],dim=0)
            valid_targets = torch.stack([rep[1:] for rep in self.raw_valid_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]

            # reconstruction
            valid_data = self.raw_valid_tup[0]
            valid_targets = self.raw_valid_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]
        valid_fitness = self.raw_valid_tup[1]

        self.train_dataset = torch.utils.data.TensorDataset(train_data, train_targets, train_fitness)
        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)
        self.valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_targets, valid_fitness)


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        print("setting up train and validation splits")
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self, shuffle_bool=True):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def valid_dataloader(self, shuffle_bool=False):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = shuffle_bool)

    def test_dataloader(self, shuffle_bool=False):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = shuffle_bool)



def str2data(dataset_name):
    """returns an uninitialized data module

    Args:
        arg ([type]): [description]

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    # model dict

    if dataset_name == 'gifford':
        data = EnsGradData

    elif dataset_name in MUT_DATASETS:
        data = MutData

    elif dataset_name == 'TAPE':
        data = TAPE

    else:
        raise NotImplementedError(f'{dataset_name} not implemented')

    return data
