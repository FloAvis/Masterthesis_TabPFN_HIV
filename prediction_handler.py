""" Helper functions for the predictions """

# Setup Imports

import numpy as np

import scipy


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

        model.fit(X_arr[train_idx], Y_arr[train_idx])

        if mode == "single":
            if method == "predict":
                y_pred_tmp = model.predict(X_arr[test_idx])

                if isinstance(y_pred_tmp, scipy.sparse.spmatrix):
                    y_pred_tmp = y_pred_tmp.todense()

            else:
                y_pred_tmp = model.predict_proba(X_arr[test_idx])

                y_pred_tmp = np.array(y_pred_tmp)
                print(y_pred_tmp.shape)

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