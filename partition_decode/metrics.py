""" Metrics to assess performance of the models as training proceeds.
"""

import numpy as np
import torch
from torch import nn

from sklearn.metrics import log_loss, brier_score_loss


"""
Common metrics
"""

# Gini impurity
def gini_impurity(P1=0, P2=0):
    denom = P1 + P2
    Ginx = 2 * (P1 / denom) * (P2 / denom)
    return Ginx


# Hellinger distance
def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
    In pure Python. Original.
    """
    return sum(
        [
            (np.sqrt(t[0]) - np.sqrt(t[1])) * (np.sqrt(t[0]) - np.sqrt(t[1]))
            for t in zip(p, q)
        ]
    ) / np.sqrt(2.0)


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
    Same as original version but without list comprehension.
    """
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))


def compute_hellinger_dist(p, q):
    """Hellinger distance between two discrete distributions.
    For Python >= 3.5 only"""
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))
    # z = np.sqrt(p) - np.sqrt(q)
    # return np.sqrt(z @ z / 2)


def compute_true_posterior(x, means=None):
    """Computes the true posterior of the Gaussian XOR"""

    if means is None:
        means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]

    mu01, mu02, mu11, mu12 = means  # [[-1, -1], [1, 1], [-1, 1], [1, -1]]

    cov = 1 * np.eye(2)
    inv_cov = np.linalg.inv(cov)

    p0 = (
        np.exp(-(x - mu01) @ inv_cov @ (x - mu01).T)
        + np.exp(-(x - mu02) @ inv_cov @ (x - mu02).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    p1 = (
        np.exp(-(x - mu11) @ inv_cov @ (x - mu11).T)
        + np.exp(-(x - mu12) @ inv_cov @ (x - mu12).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    return [p1 / (p0 + p1), p0 / (p0 + p1)]


## ECE loss
def bin_data(y, n_bins):
    """
    Partitions the data into ordered bins based on
    the probabilities. Returns the binned indices.
    """
    edges = np.linspace(0, 1, n_bins)
    bin_idx = np.digitize(y, edges, right=True)
    binned_idx = [np.where(bin_idx == i)[0] for i in range(n_bins)]

    return binned_idx


def bin_stats(y_true, y_proba, bin_idx):
    # mean accuracy within each bin
    bin_acc = [
        np.equal(np.argmax(y_proba[idx], axis=1), y_true[idx]).mean()
        if len(idx) > 0
        else 0
        for idx in bin_idx
    ]
    # mean confidence of prediction within each bin
    bin_conf = [
        np.mean(np.max(y_proba[idx], axis=1)) if len(idx) > 0 else 0 for idx in bin_idx
    ]

    return np.asarray(bin_acc), np.asarray(bin_conf)


def compute_ece_loss(y_true, y_proba, n_bins=10):
    """Computes the ECE loss"""
    bin_idx = bin_data(y_proba.max(axis=1), n_bins)
    n = len(y_true)

    bin_acc, bin_conf = bin_stats(y_true, y_proba, bin_idx)
    bin_sizes = [len(idx) for idx in bin_idx]

    ece_loss = np.sum(np.abs(bin_acc - bin_conf) * np.asarray(bin_sizes)) / n

    return ece_loss


"""
Deep Net metrics
"""

# Average stability
def compute_avg_stability(model, hybrid_set):
    """
    Computes the average stability of a model
    based on https://mlstory.org/generalization.html#algorithmic-stability
    """
    stab_dif = 0
    N = len(hybrid_set)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for i in range(N):
        model_hybrid = copy.deepcopy(model)

        ghost_loss = loss_func(model(hybrid_set[i][0]), hybrid_set[i][1])
        loss = train_model(model_hybrid, hybrid_set[i][0], hybrid_set[i][1])
        stab_dif += ghost_loss.detach().cpu().numpy().item() - loss[-1]

    return stab_dif / N


def compute_gini_mean(polytope_memberships, labels):
    """
    Compute the mean Gini impurity based on
    the polytope membership of the points and
    the labels.
    """
    gini_mean_score = []

    for l in np.unique(polytope_memberships):

        cur_l_idx = labels[polytope_memberships == l]
        pos_count = np.sum(cur_l_idx)
        neg_count = len(cur_l_idx) - pos_count
        gini = gini_impurity(pos_count, neg_count)
        gini_mean_score.append(gini)

    return np.array(gini_mean_score).mean()


def get_gini_list(polytope_memberships, labels):
    """
    Computes the Gini impurity same as compute_gini_mean
    but returns the whole list
    """
    gini_score = np.zeros(polytope_memberships.shape)

    for l in np.unique(polytope_memberships):
        idx = np.where(polytope_memberships == l)[0]
        cur_l_idx = labels[polytope_memberships == l]
        pos_count = np.sum(cur_l_idx)
        neg_count = len(cur_l_idx) - pos_count
        gini = gini_impurity(pos_count, neg_count)
        gini_score[idx] = gini

    return np.array(gini_score)


"""
Decision Forest metrics
"""


def compute_df_gini_mean(model, data, labels):
    """
    Compute the mean Gini impurity based on
    the leaf indices the data points result in and
    the labels.
    """

    leaf_idxs = model.apply(data)
    gini_mean_score = []
    for t in range(leaf_idxs.shape[1]):
        gini_arr = []
        for l in np.unique(leaf_idxs[:, t]):
            cur_l_idx = labels[leaf_idxs[:, t] == l]
            pos_count = np.sum(cur_l_idx)
            neg_count = len(cur_l_idx) - pos_count
            gini = gini_impurity(pos_count, neg_count)
            gini_arr.append(gini)

        gini_mean_score.append(np.array(gini_arr).mean())
    return np.array(gini_mean_score).mean()


"""
Matrix metrics
"""


def irm2activations(irm):
    """
    Converts an internal representation matrix to an activation matrix.
    Each activation region is a unique row in the internal representation.
    """

    regions, inverses = np.unique(irm, axis=0, return_inverse=True)
    act_mat = np.zeros((irm.shape[0], len(regions)))
    act_mat[np.arange(irm.shape[0]), inverses] = 1

    return act_mat


def fast_evals(X, pad=True, is_kernel=False):
    n = X.shape[0]
    if is_kernel:
        pass
    elif X.shape[0] < X.shape[1]:
        X = X @ X.T
    else:
        X = X.T @ X
    evals = np.linalg.svd(X, compute_uv=False)
    if pad:
        evals = np.concatenate((evals, [0]*(n-len(evals))))
    return evals


def score_matrix_representation(
    X, y=None, metric="l2", p=1, is_kernel=False, regions=False, prune_columns=False, is_evals=False
):
    """
    Scores a matrix encoding.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, p)
        Internal representation

    y : numpy.ndarray, shape (n,) (default=None)
        Targets, if needed for the metric

    metric : str (default='norm')
        Metric to apply to X, among the following choices
        - 'norm' : Lp norm of the eigenvalues
        - 'n_regions' : number of unique rows in X
        - 'entropy' : entropy of the eigenvalues
        - 'rows_mean' : mean of the rows
        - 'cols_mean' : mean of the columns

    p : int (default=1)
        Value to use for some metrics

    is_kernel : boolean (default=False)
        If True, X is a symmetric kernel matrix

    regions : boolean (default=False)
        If True, applies the metric to the regions encoded
        by the rows.

    prune_columns : boolean (default=False)
        If True, removes all columns which are constant

    is_evals : boolean (default=False)
        If True, then X is taken to be a vector of eigenvalues

    Returns
    -------
    score : float

    Notes
    -----
    If X is a region assignment matrix, the L0 norm is the number
    of regions, L1 is the number of samples, and Lp norms are
    with respect to the sizes of each region (=eigenvalues of X).
    """

    n = X.shape[0]

    if prune_columns:
        X = X[:, ~np.all(X[1:] == X[:-1], axis=0)]

    if metric in ['norm', 'entropy', 'h*']:
        if is_evals:
            evals = X
        elif regions:
            _, evals = np.unique(X, axis=0, return_counts=True)
        else:
            evals = fast_evals(X, pad=False, is_kernel=is_kernel)
        if metric == 'norm':
            score = np.sum(evals ** p) ** (1 / p)
        elif metric == 'entropy':
            evals = evals[evals > 0]
            score = (evals * np.log(evals)).sum()
        elif metric == 'h*':
            score = np.asarray([
                i / n + np.sqrt(np.sum(evals[i:]) / n) for i in range(len(evals)+1)
            ])
            score = score.argmin()
    elif metric == "n_regions":
        score = len(np.unique(X, axis=0))
    elif metric == "row_means":
        score = np.sum(np.mean(X, axis=0) ** p) ** (1 / p)
    elif metric == "col_means":
        score = np.sum(np.mean(X, axis=1) ** p) ** (1 / p)        
    else:
        raise ValueError(f'Metric {metric} is not valid')

    return score
