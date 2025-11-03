"""Utility Script for various predictive methods of TabPFN """

# Setup Imports
import pandas as pd
import numpy as np

import scipy


from sklearn.metrics import (
    precision_recall_curve,
    auc, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer


"""
Scores concerning Multilabel prediction:
"""

def prc_auc_score(y_true, y_score, multiclass="raise"):
    """
    Calculating AUC PRC for binary and multiclass setting. OVR multiclass setting
    was adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    :param y_true: True labels
    :param y_score: Predicted labels
    :param multiclass: which mode of multiclass to use
    :return: prc score
    """
    if y_score.shape[1] == 2:
        tab_prec, tab_rec, thresholds = precision_recall_curve(y_true, y_score[:, 1])
        score_prc = auc(tab_rec, tab_prec)

    else:
        if multiclass == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        elif multiclass == "ovo":
            pass
        elif multiclass == "ovr":

            label_binarizer = LabelBinarizer().fit(y_true)
            Y_test = label_binarizer.transform(y_true)

            # print(y_test)
            # print(y_onehot_test)


            n_classes = y_score.shape[1]

            # For each class
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])


            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                Y_test.ravel(), y_score.ravel()
            )

            # score_prc = average_precision_score(Y_test, y_score, average="micro")

            score_prc = auc(recall["micro"], precision["micro"])


    return score_prc


def subset_acc(y_pred, y_true, nan_mode="warning"):
    """
    Calculates the subset accuracy for multilabel prediction

    :param y_pred: Predicted labels
    :param y_true: true labels
    :param nan_mode: mode determining what happens when NaN is encountered:
                        - "warning": message informing about presence of NaNs which always leads to negative result
                        - "ignore": ignoring the NaNs in calculation
    :return: calculated subset accuracy
    """

    acc = 0

    for i in range(y_pred.shape[0]):
        tmp = 1
        for j in range(y_pred.shape[1]):
            if np.isnan(y_true.iloc[i,j]):
                if nan_mode == "ignore":
                    continue
                elif nan_mode == "warning":
                    print("Warning: True label is Missing, example not determinable")
                    tmp = 0
                    break
                else:
                    print("Invalid NaN handling")
                    return
            else:
                if y_pred.iloc[i,j] != y_true.iloc[i,j]:
                    tmp = 0
                    break

        acc = acc + tmp

    acc = acc/y_pred.shape[0]

    return acc

def exam_acc(y_pred, y_true, nan_mode="warning"):
    """
    Calculates the subset accuracy for multilabel prediction

    :param y_pred: Predicted labels
    :param y_true: true labels
    :param nan_mode: mode determining what happens when NaN is encountered:
                        - "warning": message informing about presence of NaNs which always leads to negative result
                        - "ignore": ignoring the NaNs in calculation
    :return: calculated subset accuracy
    """

    acc = 0

    for i in range(y_pred.shape[0]):
        tmp = 0
        for j in range(y_pred.shape[1]):
            if np.isnan(y_true.iloc[i,j]):
                if nan_mode == "ignore":
                    continue
                elif nan_mode == "warning":
                    print("Warning: True label is Missing, example not determinable")
                    break
                else:
                    print("Invalid NaN handling")
                    return
            else:
                if y_pred.iloc[i,j] == y_true.iloc[i,j]:
                    tmp += 1


        acc = acc + (tmp / np.sum(~np.isnan(y_true.iloc[i,:])))

    acc = acc/y_pred.shape[0]

    return acc


"""
Cross validation methods
"""

def cv_predict(model, X, Y, cv, mode="single", method="predict"):
    """
        Cross validation returning prediction probabilities of all k folds and storing them

        :param model: model used for the cross validation
        :param X: feature dataframe
        :param Y: labels dataframe
        :param cv: cross validator
        :param mode: "single" for cross validation of a single model or "ensemble" for cross validation of an ensemble method
        :param method: "predict" for return of labels or "predict_proba" for the return of prediction probabilities
        :return:    y_pred: predicted probabilities or labels of examples in X in shape (T, L)
                    y_true: true labels of examples in X in shape (T, L)
        """
    if mode not in ["single", "ensemble"]:
        raise Exception("Mode not valid. Please Select 'single' for normal estimators or 'ensemble' for ensembles")

    if method not in ["predict", "predict_proba"]:
        raise Exception("Method not valid. Please Select 'predict' for label prediction or 'predict_proba' for prediction probabilities")

    if method == "predict":
        if mode == "single":
            y_pred = np.zeros((X.shape[0], Y.shape[1]))
            y_true = np.zeros((X.shape[0], Y.shape[1]))# (n_samples, n_labels, n_classes)
        elif mode == "ensemble":
            y_pred = np.zeros((model.n_jobs, X.shape[0], Y.shape[1]))  # (n_samples, n_labels, n_classes)
            y_true = np.zeros((model.n_jobs, X.shape[0], Y.shape[1]))  # (n_samples, n_labels, n_classes)

    #print(method)

    elif method == "predict_proba":
        if mode == "single":
            y_pred = np.zeros((X.shape[0], Y.shape[1], 2))
            y_true = np.zeros((X.shape[0], Y.shape[1]))  # (n_samples, n_labels, n_classes)
        elif mode == "ensemble":
            y_pred = np.zeros((model.n_jobs, X.shape[0], Y.shape[1], 2))  # (n_jobs, n_samples, n_labels, n_classes)
            y_true = np.zeros((model.n_jobs, X.shape[0], Y.shape[1]))  # (n_jobs, n_samples, n_labels, n_classes)



    X_arr = np.array(X)
    Y_arr = np.array(Y)

    counter = 0
    for train_idx, test_idx in cv.split(X):
        counter += 1
        print("CV {}".format(counter))

        model.fit(pd.DataFrame(X_arr[train_idx], columns=list(X.columns.values())), pd.DataFrame(Y_arr[train_idx], list(Y.columns.values())))

        if mode == "single":
            if method == "predict":
                y_pred_tmp = model.predict(X_arr[test_idx])

                if isinstance(y_pred_tmp, scipy.sparse.spmatrix):
                    y_pred_tmp = y_pred_tmp.todense()

            else:
                y_pred_tmp = model.predict_proba(X_arr[test_idx])

            if method == "predict_proba" and len(y_pred_tmp.shape) == 2:
                y_pred_tmp_tmp = np.zeros((y_pred_tmp.shape[0], y_pred_tmp.shape[1], 2))
                y_pred_tmp_tmp[:,:,1] = y_pred_tmp
                y_pred_tmp = y_pred_tmp_tmp


            if y_pred[test_idx].shape != np.array(y_pred_tmp).shape:

                y_pred[test_idx] = np.stack(y_pred_tmp, axis=1)
            else:
                y_pred[test_idx] = y_pred_tmp
            y_true[test_idx] = Y_arr[test_idx]
        else:

            if method == "predict":
                y_pred_tmp = model.predict(X_arr[test_idx])

            else:
                y_pred_tmp = model.predict_proba(X_arr[test_idx])

            if y_pred[:,test_idx].shape != np.array(y_pred_tmp).shape:
                y_pred[:, test_idx] = np.stack(y_pred_tmp, axis=1)
            else:
                y_pred[:,test_idx] = model.predict_proba(X_arr[test_idx])
                y_true[:,test_idx] = Y_arr[test_idx]

    #if method == "predict_proba":
    #    y_pred = y_pred[..., 1]

    return y_pred, y_true