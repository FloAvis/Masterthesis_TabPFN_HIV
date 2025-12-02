"""Functions for saving of results and metrics for evaluations """

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_curve
)
from sklearn.preprocessing import LabelBinarizer

from pathlib import Path



def get_kfold(kf, X, Y):
    """
    get the k fold partition of the dataset

    :param kf: KFold partitioner
    :param X: feature DataFrame
    :param Y: target DataFrame
    :return: 1d array of k folds
    """
    kfolds = np.zeros((Y.shape[0], 1))

    k = 0

    for _, test in kf.split(X, Y):
        for i in test:
            kfolds[i] = k
        k += 1

    return kfolds

def calc_labels(y_pred):
    """
    Returns the labels for a given probability matrix

    :param y_pred_probas: Probabilities of predictions
    :return: np.ndarray of labels
    """

    y_pred_probas = np.array(y_pred)


    y_pred_new = np.zeros((y_pred_probas[0].shape[0], len(y_pred_probas)))

    #print(y_pred_new)

    for j, clas in enumerate(y_pred_probas):
        for i in range(clas.shape[0]):
            if clas[i, 1] >= 0.5:
                y_pred_new[i, j] = 1
            else:
                y_pred_new[i, j] = 0

    return y_pred_new

def save_multilabel(y_pred, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction results to a file for later evaluation

    :param y_pred: Predicted labels
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted and true labels into file
    """

    if y_true.shape != y_pred.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(y_pred.shape))

    splt = label.split('/')[:-1]
    sub_filepath = '/'.join(splt)

    #print(sub_filepath)

    Path(path + sub_filepath).mkdir(parents=True, exist_ok=True)

    y_pred_new = y_pred.add_prefix("Pred_")

    if k_folds is not None:
        y_pred_new["kFolds"] = k_folds

    y_true_new = y_true.add_prefix("True_")

    y_true_new.reset_index(inplace=True, drop=True)

    df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

    df.to_csv(path + label + ".csv")

def save_multilabel_proba(y_pred_probas, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    y_pred_dict = {}

    drugs = y_true.columns.values.tolist()

    print(drugs)
    print(y_pred_probas.shape)

    for i, probas in enumerate(y_pred_probas):
        y_pred_dict.update({drugs[i]: probas[:, 1]})

    y_pred = pd.DataFrame(y_pred_dict)


    if y_true.shape != y_pred.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(y_pred.shape))


    splt = label.split('/')[:-1]
    sub_filepath = '/'.join(splt)

    #print(sub_filepath)

    Path(path + sub_filepath).mkdir(parents=True, exist_ok=True)

    y_pred_new = y_pred.add_prefix("Pred_Proba_")

    if k_folds is not None:
        y_pred_new["kFolds"] = k_folds


    y_true_new = y_true.add_prefix("True_")

    y_true_new.reset_index(inplace=True, drop=True)

    df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

    df.to_csv(path + label + ".csv")


def save_ensemble(y_pred_ensemble, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    if np.array(y_pred_ensemble).shape[1:3] != y_true.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(np.array(y_pred_probas_ensemble).T.shape[1:3]))


    Path(path + label).mkdir(parents=True, exist_ok=True)

    drugs = y_true.columns.values.tolist()

    for j, y_pred_arr in enumerate(y_pred_ensemble):

        y_pred = pd.DataFrame(y_pred_arr, columns=drugs)

        y_pred_new = y_pred.add_prefix("Pred_")

        if k_folds is not None:
            y_pred_new["kFolds"] = k_folds

        y_true_new = y_true.add_prefix("True_")

        y_true_new.reset_index(inplace=True, drop=True)

        df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

        df.to_csv(path + label + "/Ensemble_" + str(j) + ".csv")



def save_ensemble_proba(y_pred_probas_ensemble, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    if np.array(y_pred_probas_ensemble).T.shape[1:3] != y_true.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(np.array(y_pred_probas_ensemble).T.shape[1:3]))


    Path(path + label).mkdir(parents=True, exist_ok=True)

    for j, y_pred_probas in enumerate(y_pred_probas_ensemble):
        y_pred_dict = {}

        drugs = y_true.columns.values.tolist()

        for i, probas in enumerate(y_pred_probas):
            y_pred_dict.update({drugs[i]: probas[:, 1]})

        y_pred = pd.DataFrame(y_pred_dict)

        y_pred_new = y_pred.add_prefix("Pred_Proba_")

        if k_folds is not None:
            y_pred_new["kFolds"] = k_folds


        y_true_new = y_true.add_prefix("True_")

        y_true_new.reset_index(inplace=True, drop=True)

        df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

        df.to_csv(path + label + "/Ensemble_probas_" + str(j) + ".csv")


"""
Scores concerning Multilabel prediction:
"""

def calc_metrics(paths, models, metric, metric_args, ending="", drop_na=True, return_groups=False):
    """
    Calculating AUC PRC for binary and multiclass setting. OVR multiclass setting
    was adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    :param y_true: True labels
    :param y_score: Predicted labels
    :param multiclass: which mode of multiclass to use
    :return: prc score
    """
    means = {}
    stds = {}

    for path in paths:

        acc_list_mean = []
        acc_list_std = []


        for model in models:
            results = pd.read_csv(path + model + ending)

            if drop_na:
                results.dropna(subset=results.columns[results.columns.str.startswith('True_')].tolist(), inplace=True)

            subs_accs_groups = results.groupby(by="kFolds").apply(
                lambda x: metric(x.filter(regex="True_*"), x.filter(regex="Pred_*"), **metric_args),
                include_groups=False)

            if return_groups:
                acc_list_mean.append(list(subs_accs_groups))
            else:
                acc_list_mean.append(subs_accs_groups.mean())
                acc_list_std.append(subs_accs_groups.std())

        means.update({path.split("/")[-1].strip("_"): acc_list_mean})
        if not return_groups:
            stds.update({path.split("/")[-1].strip("_"): acc_list_std})


    if return_groups:
        return means
    else:
        return means, stds


def prc_auc_score(y_true, y_score, multiclass="raise"):
    """
    Calculating AUC PRC for binary and multiclass setting. OVR multiclass setting
    was adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    :param y_true: True labels
    :param y_score: Predicted labels
    :param multiclass: which mode of multiclass to use
    :return: prc score
    """

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if y_score.shape[-1] == 2:
        tab_prec, tab_rec, thresholds = precision_recall_curve(y_true, y_score[:, 1])
        score_prc = auc(tab_rec, tab_prec)

    else:
        if multiclass == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        elif multiclass == "ovo":
            n_classes = y_score.shape[1]

            # For each label
            average_precision = []
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
                average_precision.append(auc(recall, precision))

            # score_prc = average_precision_score(Y_test, y_score, average="micro")

            score_prc = np.mean(average_precision)
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


def subset_acc(y_true, y_pred, nan_mode="warning"):
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

def exam_acc(y_true, y_pred, nan_mode="warning"):
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

