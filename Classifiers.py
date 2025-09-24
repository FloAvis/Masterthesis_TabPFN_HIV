"""File for implementation of different classifiers for multilabel prediction with TabPFN"""

import numpy as np
import pandas as pd
import random
from itertools import permutations
import collections
import warnings
from tabpfn import TabPFNClassifier

import utils

from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryRelevance(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, random_state=42, use_labels=False):
        # Store parameters
        self.random_state = random_state
        self.estimator = estimator
        self.use_labels = use_labels

        # Initialize the underlying classifier with given parameters
        #self.clf = self.estimator(**tabpfn_params)



    def fit(self, X, Y, sample_weight=None, **fit_params):

        self.estimators_ = []

        if isinstance(X, np.ndarray):
            col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
            df_X = pd.DataFrame(X, columns=col_names)
        else:
            df_X = X

        if isinstance(Y, np.ndarray):
            col_names = ["Y_" + str(s) for s in list(range(Y.shape[1]))]
            df_Y = pd.DataFrame(Y, columns=col_names)
        else:
            df_Y = Y

        for i in range(Y.shape[1]):


            tmp_comb = df_X.join(df_Y)

            tmp_comb.dropna(subset=df_Y.columns.values.tolist()[i], inplace=True)

            if self.use_labels:
                filt_X = tmp_comb.drop(df_Y.columns.values.tolist()[i], axis=1)

                filt_Y = tmp_comb[df_Y.columns.values.tolist()[i]]

                filt_y = np.asarray(filt_Y)

                self.estimators_.append(self.estimator(random_state=self.random_state).fit(filt_X, filt_y[:, 0]))

            else:
                filt_X = tmp_comb[df_X.columns.values.tolist()]

                filt_Y = tmp_comb[df_Y.columns.values.tolist()]

                filt_y = np.asarray(filt_Y)

                self.estimators_.append(self.estimator(random_state=self.random_state).fit(filt_X, filt_y[:, i]))

        self.classes_ = [estimator.classes_ for estimator in self.estimators_]

        return self


    def predict(self, X, Y=None):

        y = []

        if self.use_labels:

            if Y == None:
                raise Exception("Labels not given for prediction")
            else:

                if isinstance(X, np.ndarray):
                    col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
                    df_X = pd.DataFrame(X, columns=col_names)
                else:
                    df_X = X

                if isinstance(Y, np.ndarray):
                    col_names = ["Y_" + str(s) for s in list(range(Y.shape[1]))]
                    df_Y = pd.DataFrame(Y, columns=col_names)
                else:
                    df_Y = Y


                tmp_comb = df_X.join(df_Y)

                for i, e in enumerate(self.estimators_):
                    tmp_X = tmp_comb.drop(df_Y.columns.values.tolist()[i], axis=1)
                    y.append(e.predict(tmp_X))
        else:

            for e in self.estimators_:
                y.append(e.predict(X))


        return np.asarray(y).T

    def predict_proba(self, X, Y=None):

        if self.use_labels:

            if Y == None:
                raise Exception("Labels not given for prediction")
            else:

                results = []

                if isinstance(X, np.ndarray):
                    col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
                    df_X = pd.DataFrame(X, columns=col_names)
                else:
                    df_X = X

                if isinstance(Y, np.ndarray):
                    col_names = ["Y_" + str(s) for s in list(range(Y.shape[1]))]
                    df_Y = pd.DataFrame(Y, columns=col_names)
                else:
                    df_Y = Y

                tmp_comb = df_X.join(df_Y)

                for i, e in enumerate(self.estimators_):
                    tmp_X = tmp_comb.drop(df_Y.columns.values.tolist()[i], axis=1)

                    results.append(e.predict_proba(tmp_X))

        else:

            results = [estimator.predict_proba(X) for estimator in self.estimators_]

        return results



class ClassifierChains(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, random_state=42, order="random"):
        # Store parameters
        self.random_state = random_state
        self.estimator = estimator
        self.order = order

        # Initialize the underlying classifier with given parameters
        #self.clf = self.estimator(**tabpfn_params)



    def fit(self, X, Y, sample_weight=None, **fit_params):


        if isinstance(self.order, collections.abc.Sequence) and not isinstance(self.order, str) and len(self.order) == Y.shape[1]:
            order = self.order
        else:
            if self.order != "random":
                warnings.warn("Invalid ordering, contunuing with random")
            random.seed(self.random_state)

            order = list(range(Y.shape[1]))

            random.shuffle(order)

            self.order = order

        #print(order)

        self.estimators_ = []


        if isinstance(X, np.ndarray):
            col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
            df_X = pd.DataFrame(X, columns=col_names)
        else:
            df_X = X

        if isinstance(Y, np.ndarray):
            col_names = ["Y_" + str(s) for s in list(range(Y.shape[1]))]
            df_Y = pd.DataFrame(Y, columns=col_names)

            # currently only works with binary due to cross_val_predict changin Nan to a number higher than number of classes i presume

            df_Y.replace(2, np.nan, inplace=True)
        else:
            df_Y = Y

            #currently only works with binary due to cross_val_predict changin Nan to a number higher than number of classes i presume
            df_Y.replace(2, np.nan, inplace=True)


        for est_num, i in enumerate(self.order):

            tmp_X = df_X.copy()


            tmp_comb = tmp_X.join(df_Y)

            tmp_comb.dropna(subset=df_Y.columns.values.tolist()[i], inplace=True)

            filt_X = tmp_comb[df_X.columns.values.tolist()]

            filt_Y = tmp_comb[df_Y.columns.values.tolist()]

            filt_y = np.asarray(filt_Y)


            for j in range(est_num):
                filt_X[("Feat_" + str(j))] = filt_y[:, self.order[j]]


            self.estimators_.append(self.estimator(random_state=self.random_state).fit(filt_X, filt_y[:, i]))

        self.classes_ = [estimator.classes_ for estimator in self.estimators_]

        return


    def predict(self, X):

        if isinstance(X, np.ndarray):
            col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
            df_X = pd.DataFrame(X, columns=col_names)
        else:
            df_X = X


        y = np.zeros((len(self.estimators_), X.shape[0]))

        tmp_X = df_X.copy()

        for est_num, i in enumerate(self.order):

            if est_num != 0:
                tmp_X[("Feat_" + str(est_num - 1))] = y[self.order[est_num - 1]]

            y[i] = self.estimators_[est_num].predict(tmp_X)

        return y.T

    def predict_proba(self, X):

        if isinstance(X, np.ndarray):
            col_names = ["X_" + str(s) for s in list(range(X.shape[1]))]
            df_X = pd.DataFrame(X, columns=col_names)
        else:
            df_X = X

        tmp_X = df_X.copy()

        results = [None] * len(self.estimators_)

        for est_num, i in enumerate(self.order):

            if est_num != 0:
                tmp_X[("Feat_" + str(est_num - 1))] = y_pred_class

            results[i] = self.estimators_[est_num].predict_proba(tmp_X)

            y_pred_class = np.argmax(results[i], axis=1)

        return np.stack(results, axis=1)

class Ensemble(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_jobs=5, random_state=42, averaging=False):
        # Store parameters
        self.random_state = random_state
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.averaging = averaging

        # Initialize the underlying classifier with given parameters
        #self.clf = self.estimator(**tabpfn_params)



    def fit(self, X, Y):

        perms = list(permutations(range(Y.shape[1])))

        if self.n_jobs > len(perms):
            warnings.warn(f"# of jobs greater than # of possible combinations, reduction to # of combinations: {self.n_jobs} > {len(perms)}")
            orders = perms
        else:
            random.seed(self.random_state)

            orders = random.sample(perms, self.n_jobs)


        self.chains_ = [self.estimator(TabPFNClassifier, random_state=self.random_state, order=order) for order in orders]
        for chain in self.chains_:
            chain.fit(X, Y)

        self.classes_ = [chain.classes_ for chain in self.chains_]

        return self


    def predict(self, X):

        if self.averaging:
            return (np.median(np.array([chain.predict(X) for chain in self.chains_]), axis=0) > 0.5) * 1

        else:
            return np.array([chain.predict(X) for chain in self.chains_])



    def predict_proba(self, X):


        if self.averaging:
            return np.mean(np.array([chain.predict_proba(X) for chain in self.chains_])[...,1], axis=0)
            #return np.mean(np.array([chain.predict_proba(X) for chain in self.chains_]), axis=0)

        else:
            return np.array([chain.predict_proba(X) for chain in self.chains_])