"""Implementation"""

# Setup Imports
import pandas as pd
import numpy as np
import time
import os

import torch
from DELA.DELAModel import DELAModel
from DELA.utils import init_random_seed, generate_default_config, clear_old_logs
from DELA.dataset import DatasetLoader, Dataset, PI, INI, NRTI, NNRTI
import prediction_handler
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.preprocessing import OneHotEncoder


import scipy

import DELA as dela


# Baseline Imports

from tabpfn import TabPFNClassifier

from Classifiers import ClassifierChains as cc



def main():
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",r"../data/NNRTI_DataSet.txt"]
    #files = [r"../data/INI_DataSet.txt"]

    for file in files:

        # Reading in and processing high quality File
        df = pd.read_csv(file, sep='\t')

        # removing index and summary column
        df = df.iloc[:, 1:-1]

        # list of current drugs of the dataset
        drugs = [drug for drug in list(df.columns) if not drug.startswith("P")]

        # Filtering out drugs with less than 10 labels present
        unusable_drugs = [drug for drug in drugs if df[drug].count() <= 10]

        if len(unusable_drugs) > 0:
            df.drop(columns=unusable_drugs, inplace=True)

            drugs = [drug for drug in drugs if drug not in unusable_drugs]

        # dropping rows with na labels
        df.dropna(subset=drugs, inplace=True)

        enc = OneHotEncoder(handle_unknown='error')

        X = df.drop(drugs, axis=1)

        enc.fit(X)
        X_trafo = enc.transform(X).toarray()
        Y = utils.get_classes(df, drugs, mode="binary")

        # clf = TabPFNClassifier()

        multi_target_pfn = cc(TabPFNClassifier, random_state=42)

        use_kfold = True

        folds = 5

        n_jobs = 4

        X_train, X_test, y_train, y_test = train_test_split(X_trafo, Y, test_size=0.33, random_state=42)

        # Setting configurations
        configs = generate_default_config()
        # device params
        configs['use_gpu'] = True
        configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
        # training params

        configs['beta'] = 1e-4

        # Loading dataset
        configs['shuffle'] = True

        configs['data_standardizing'] = False

        #dataset = Dataset()
        configs['nfold'] = 5

        # Setting architecture params
        configs['model_name'] = 'DELAModel'


        #dataset = eval(file.split("/")[-1].split("_")[0])(configs=configs, X=X_train, y=y_train, nfold=configs['nfold'])
        configs['dataset_name'] = file.split("/")[-1].split("_")[0]
        print(configs['dataset_name'])


        # Setting architecture params
        configs['model_name'] = 'DELAModel'
        #configs['in_features'] = dataset.feat_dim
        #configs['num_classes'] = dataset.num_class
        configs['latent_dim'] = 50

        # Setting other params
        configs['exp'] = "1"
        configs['exp_dir'] = os.path.join(configs['model_name'],
                                          configs['exp'],
                                          configs['dataset_name'])
        configs['save_checkpoint_path'] = os.path.join(configs['exp_dir'], 'checkpoint')

        # ensemble = en(cc, random_state=42, n_jobs=n_jobs)

        if not use_kfold:

            dataset = eval(file.split("/")[-1].split("_")[0])(configs=configs, X=X_train, y=y_train,
                                                              nfold=configs['nfold'])

            configs['in_features'] = dataset.feat_dim
            configs['num_classes'] = dataset.num_class


            model = DELAModel(configs)
            count = 1

            dataset.data_splitter()

            print(type(dataset.train_dataloader.data_inds))
            print(dataset.test_dataloader.data_inds)

            #print()


            train_dataloader = dataset.train_dataloader
            val_dataloader = dataset.val_dataloader
            test_dataloader = dataset.test_dataloader

            # Training with evaluation
            model.reset_parameters()
            model.configs['best_epoch'] = 0
            model.train(train_dataloader, val_dataloader, quiet_mode=False)
            model.load_checkpoint(model.configs['best_checkpoint_path'])
            model.configs['start_epoch'] = 0

            y_pred, y_pred_probas = model.predict(torch.from_numpy(np.array(X_test).astype(float)).type(configs['dtype']))


            #print(np.array(y_pred).shape)

            #print(np.array(y_pred_probas).shape)

            if isinstance(y_pred, scipy.sparse._csr.csr_matrix):
                y_pred = y_pred.todense()

            y_pred_df = pd.DataFrame(y_pred, columns=drugs)

            y_test_df = pd.DataFrame(y_test, columns=drugs)

            y_pred_probas_df = pd.DataFrame(y_pred_probas, columns=drugs)

            # print(np.array(y_pred_proba).shape)

            utils.save_multilabel(y_pred_df, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                0] + "_DELA"))

            utils.save_multilabel(y_pred_probas_df, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                0] + "_DELA_probas"))
        

        else:
            dataset = eval(file.split("/")[-1].split("_")[0])(configs=configs, X=X_trafo, y=Y,
                                                              nfold=configs['nfold'])

            configs['in_features'] = dataset.feat_dim
            configs['num_classes'] = dataset.num_class

            model = DELAModel(configs)

            y_pred_con = []

            y_pred_probas_con = []

            y_test_con = []

            kfolds = []

            for count in range(1, configs['nfold']+1):

                dataset.data_cv_splitter(count, configs['nfold'], configs['shuffle'], configs['rand_seed'])

                train_dataloader = dataset.train_dataloader
                val_dataloader = dataset.val_dataloader
                test_dataloader = dataset.test_dataloader

                # Training with evaluation
                model.reset_parameters()
                model.configs['best_epoch'] = 0
                model.train(train_dataloader, val_dataloader, quiet_mode=False)
                model.load_checkpoint(model.configs['best_checkpoint_path'])
                model.configs['start_epoch'] = 0



                y_pred, y_pred_probas, y_true = model.test_labels(test_dataloader)

                y_pred_con.append(y_pred)

                y_pred_probas_con.append(y_pred_probas)

                y_test_con.append(y_true)

                kfolds += [count-1] * y_pred.shape[0]

                # if isinstance(y_pred, scipy.sparse._csr.csr_matrix):
                #    y_pred = y_pred.todense()

            kfolds = np.array(kfolds)

            y_pred_df = pd.DataFrame(np.concatenate(y_pred_con), columns=drugs)

            y_test_df = pd.DataFrame(np.concatenate(y_test_con), columns=drugs)

            y_pred_probas_df = pd.DataFrame(np.concatenate(y_pred_probas_con), columns=drugs)

            # y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

            # changed the saving mechanism of classifier chain, new way is better but I don't wanna change my system so gotta convert back again
            # y_pred_new = np.stack(y_pred_new, axis=1)

            # print(y_pred_new.shape)

            # y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)
            """
            kfolds = np.zeros((y_pred.shape[0], 1))

            k = 0

            for _, test in kf.split(X, Y):
                for i in test:
                    kfolds[i] = k
                k += 1
            """

            # print(np.array(y_pred_proba).shape)

            utils.save_multilabel(y_pred_df, y_test_df, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                0] + "_DELA_" + str(folds) + "_fold"))

            utils.save_multilabel(y_pred_probas_df, y_test_df, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                0] + "_DELA_probas" + "_" + str(folds) + "_fold"))




if __name__ == '__main__':
    main()