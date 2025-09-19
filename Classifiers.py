"""File for implementation of different classifiers for multilabel prediction with TabPFN"""

import numpy as np
import pandas as pd
import random

import utils

from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryRelevance(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, random_state=42):
        # Store parameters
        self.random_state = random_state
        self.estimator = estimator

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

            filt_X = tmp_comb[df_X.columns.values.tolist()]

            filt_Y = tmp_comb[df_Y.columns.values.tolist()]

            filt_y = np.asarray(filt_Y)

            self.estimators_.append(self.estimator(random_state=self.random_state).fit(filt_X, filt_y[:, i]))

        self.classes_ = [estimator.classes_ for estimator in self.estimators_]

        return self


    def predict(self, X):

        y = []

        for e in self.estimators_:
            y.append(e.predict(X))


        return np.asarray(y).T

    def predict_proba(self, X):

        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results



class ClassifierChains(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, random_state=42):
        # Store parameters
        self.random_state = random_state
        self.estimator = estimator


        # Initialize the underlying classifier with given parameters
        #self.clf = self.estimator(**tabpfn_params)



    def fit(self, X, Y, sample_weight=None, **fit_params):

        print(self.random_state)
        random.seed(self.random_state)

        order = list(range(Y.shape[1]))
        random.shuffle(order)

        self.order = order

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

        return self


    def predict(self, X):

        y = np.zeros((len(self.estimators_), X.shape[0]))

        tmp_X = X.copy()

        for est_num, i in enumerate(self.order):

            if est_num != 0:
                tmp_X[("Feat_" + str(est_num - 1))] = y[self.order[est_num - 1]]

            y[i] = self.estimators_[est_num].predict(tmp_X)

        return y.T

    def predict_proba(self, X):

        tmp_X = X.copy()

        results = [None] * len(self.estimators_)

        for est_num, i in enumerate(self.order):

            if est_num != 0:
                tmp_X[("Feat_" + str(est_num - 1))] = y_pred_class

            results[i] = self.estimators_[est_num].predict_proba(tmp_X)

            y_pred_class = np.argmax(results[i], axis=1)

        return results