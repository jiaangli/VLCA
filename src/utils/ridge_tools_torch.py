# taken from https://github.com/mtoneva/brain_language_nlp/blob/master/utils/ridge_tools.py
# with some modifications using PyTorch to accelerate.

import numpy as np
import torch
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"


# error functions
def corr_torch(X_tensor, Y_tensor):
    X_zscore = (X_tensor - torch.mean(X_tensor, dim=0)) / torch.std(X_tensor, dim=0)
    Y_zscore = (Y_tensor - torch.mean(Y_tensor, dim=0)) / torch.std(Y_tensor, dim=0)
    return torch.mean(X_zscore * Y_zscore, dim=0)


def R2_torch(Pred, Real):
    SSres = torch.mean((Real - Pred) ** 2, 0)
    SStot = torch.var(Real, 0)
    return torch.nan_to_num(1 - SSres / SStot)


def ridge_torch(X_torch, Y_torch, lmbda):
    # X_torch = torch.from_numpy(X).to(device)
    # Y_torch = torch.from_numpy(Y).to(device)
    # Calculate ridge regression coefficients using PyTorch functions
    # I = torch.eye(X_torch.shape[1]).to(device)
    # coeffs = torch.inverse(X_torch.transpose(0, 1) @ X_torch + lmbda * I) @ X_torch.transpose(0, 1) @ Y_torch
    I = torch.eye(X_torch.shape[1], device=X_torch.device)
    # coeffs = torch.empty(X_torch.size(1), X_torch.size(1), device=X_torch.device)
    coeffs = torch.inverse(X_torch.transpose(0, 1) @ X_torch + lmbda * I)
    coeffs = coeffs @ X_torch.transpose(0, 1) @ Y_torch
    # Move the coefficients back to CPU and convert to numpy array
    return coeffs


def ridge_by_lambda_torch(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    # x,y,xval,yval are the same thing just split into training and testing is to calculate the errors
    #
    error = torch.zeros((lambdas.shape[0], Y.shape[1])).to(device)
    # for every lambda calculate an error
    # lambdas are an array of values
    for idx, lmbda in enumerate(lambdas):
        # weights = ridge(X, Y, lmbda)
        weights = ridge_torch(X, Y, lmbda)
        # error for every lambda is calculated
        # 1-R2
        # get back an error for every lambda
        # np.dot(Xval,weights) these are the predictions. Essential is the question if we combine the model weights with the extracted features from testing can we predict
        # accurately the fMRI recordings
        # Yval these are the labels
        # error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
        error[idx] = 1 - R2_torch(Xval @ weights, Yval)
    return error


def kernel_ridge_torch(X_torch, Y_torch, lmbda):
    # so the kernel ridge regression is defined as
    # xTranspose times X times inverted(XTranspose times X + lambda times identity matrix) times Y
    # X.T is the first transpose XTranspose
    # inv is the inverted product
    # X.dot(X.T) is the X times X transpose
    # np.eye is the identity matrix
    # lmbda * np.eye  lambda times identity matrix
    # so the whole thing gives us the ridge regression with kernel
    # I = torch.eye(X_torch.shape[0], device=X_torch.device)
    # coeffs = torch.empty(X_torch.size(1), X_torch.size(1), device=X_torch.device)
    coeffs = torch.inverse(
        X_torch @ X_torch.transpose(0, 1) + lmbda * torch.eye(X_torch.shape[0], device=X_torch.device))
    coeffs = X_torch.transpose(0, 1) @ coeffs @ Y_torch
    # return np.dot(X.T.dot(inv(X.dot(X.T) + lmbda * np.eye(X.shape[0]))), Y)
    return coeffs


def kernel_ridge_by_lambda_torch(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = torch.zeros((lambdas.shape[0], Y.shape[1])).to(device)
    for idx, lmbda in enumerate(lambdas):
        weights = kernel_ridge_torch(X, Y, lmbda)
        error[idx] = 1 - R2_torch(Xval @ weights, Yval)
    return error


# main loop that utilises the above functions
def cross_val_ridge(train_features, train_data, n_splits=10,
                    lambdas=np.array([10 ** i for i in range(-6, 10)]),
                    method='plain',
                    do_plot=False):
    ridge_1 = dict(plain=ridge_by_lambda_torch,
                   kernel_ridge=kernel_ridge_by_lambda_torch)[method]
    ridge_2 = dict(plain=ridge_torch,
                   kernel_ridge=kernel_ridge_torch)[method]

    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = torch.zeros((nL, train_data.shape[1])).to(device)

    kf = KFold(n_splits=n_splits)
    # start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        # print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn], train_data[trn],
                       train_features[val], train_data[val],
                       lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost, aspect='auto')
        # get the cost for every lambda value
        r_cv += cost
        # if icv%3 ==0:
        #    print(icv)
        # print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv, aspect='auto', cmap='RdBu_r')

    # get the best run index (lambda) for every every target features 
    # (in brain is voxels, for NLP respresentations is every dimension of the representation)
    argmin_lambda = np.argmin(r_cv.cpu().numpy(), axis=0) 
    weights = torch.zeros((train_features.shape[1], train_data.shape[1])).to(device)
    for idx_lambda in range(lambdas.shape[0]):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda # idx_vox is a boolean array, vox here means target features dimension
        # get the best possible prediction for a different lambda
        if any(idx_vox):
            weights[:, idx_vox] = ridge_2(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])