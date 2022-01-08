import numpy as np
import scipy
from scipy import stats

import scipy.linalg as la
from numpy.linalg import matrix_rank, norm

import pandas as pd

import copy


def ger_matrix_from_poly(model, dataset, poly_m):
    # L_matrices = {'0/1': [], 'true_label':[], 'est_label':[], 'est_poster':[]}
    L_matrices = []
    test_y, pred_y, test_acc = get_label_pred(model, dataset)
    print(pred_y.shape, poly_m.shape)
    unique_poly = np.unique(poly_m)
    n_poly = len(unique_poly)

    # for key in L_matrices:
    L_mat = np.zeros((len(poly_m), n_poly))
    for idx, poly_i in enumerate(poly_m):
        poly_idx = np.where(unique_poly == poly_i)
        L_mat[idx, poly_idx] = pred_y[idx] + 1
        # if key == '0/1':
        #     L_mat[idx, poly_idx] = pred_label[idx]
        # elif key == 'true_label':
        #     L_mat[idx, poly_idx] = 2*y_train[idx]-1
        # elif key == 'est_label':
        #     L_mat[idx, poly_idx] = 2*pred_label[idx]-1
        # elif key == 'est_poster':
        #     L_mat[idx, poly_idx] = 2*pred_poster[idx]-1
        # L_matrices[key].append(L_mat)

    # gen_gap = abs((1-test_acc) - (1-train_acc))
    test_gen_err = 1 - test_acc
    return np.array(L_mat), test_gen_err


def get_label_pred(model, dataset, computeOver=500, batchSize=50):

    it = iter(dataset.repeat(-1).shuffle(50000, seed=1).batch(batchSize))
    N = computeOver // batchSize
    batches = [next(it) for i in range(N)]

    test_y = [batch[1] for batch in batches]

    # ds = dataset.repeat(-1).shuffle(50000, seed=1).batch(batchSize)
    # preds = model.predict(x=ds, steps=N, verbose=False)
    # print(preds.shape)
    # preds = model.predict(x=dataset)
    # print(preds.shape)
    # pred_y = np.argmax(preds, axis=-1)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    acc, size = 0, 0
    y_true = []
    preds = []
    for batch in batches:
        test_loss, test_acc = model.evaluate(batch[0], batch[1], verbose=False)
        acc += test_acc * len(batch[1])
        size += len(batch[1])
        preds.extend(model.predict(batch[0]))
    acc = acc / size
    pred_y = np.argmax(preds, axis=-1)
    print(pred_y.shape)
    return test_y, pred_y, acc


##********** Matrix ranks *************##
def get_stable_rank(m):
    """
    Compute stable rank of a matrix: frobenius norm (squared) / spectral norm (squared)
    """
    return norm(m, ord="fro") ** 2 / norm(m, ord=2) ** 2


def get_KF_Schatten_norms(m, num_k=5):
    """
    Compute different matrix norms
    Input: m - 2d matrix (n by L)
    Return: 4 1D numpy arrays
    - First: Ky-Fan results [Un-normalized], where k-th element is the sum of top-k singular values
    - Second: Ky-Fan results [Normalized], where k-th element is the ratio of the variance explained by top-k singular values
    - Third: Ky-Fan results on m^T @ m, where k-th element is the sum of top-k eigenvalues of m^T @ m (i.e., singular values of (m) squared)
    - Fourth: Schatten results, where k-th element is the k-norm, (sum_i sigma^k)^{1/k}
    """
    ss = np.linalg.svd(m, full_matrices=False, compute_uv=False)
    KFs_raw = np.array([ss[: i + 1].sum() for i in range(num_k)])
    total = ss.sum()
    KFs = KFs_raw / total
    evalues = ss ** 2
    KFs_kernel = np.array([evalues[: i + 1].sum() for i in range(num_k)])
    Schattens = [total]
    ss_pow = copy.deepcopy(ss)
    for i in range(2, num_k + 1):
        ss_pow = np.multiply(ss_pow, ss)
        Schattens.append(np.power(ss_pow.sum(), 1 / i))
    return KFs_raw, KFs, KFs_kernel, np.array(Schattens)


def graph_metrics(m):
    """
    Input: internal representation, n by L
    Return: 2-tuple
    - clustering coefficients of a bipartite graph built from m, a measure of local density of the connectivity
    ref: https://networkx.org/documentation/stable//reference/algorithms/generated/networkx.algorithms.bipartite.cluster.clustering.html#networkx.algorithms.bipartite.cluster.clustering
    - modularity: relative density of edges inside communities with respect to edges outside communities.
    ref: https://python-louvain.readthedocs.io/en/latest/api.html#community.modularity
    """
    from community import modularity
    from community import community_louvain
    from networkx.algorithms import bipartite
    sM = scipy.sparse.csr_matrix(m)
    G = bipartite.matrix.from_biadjacency_matrix(sM)
    avg_c = bipartite.average_clustering(G, mode="dot")
    partition = community_louvain.best_partition(G)
    modularity = modularity(partition, G)

    return avg_c, modularity


def compute_complexity(L, k=5, from_evalues=False, from_gram=False):
    """
    Computes a variety of internal representation complexity metrics at once.
    Parameters
    ----------
    L : numpy.ndarray, shape (n_samples, ...)
        internal representation matrix or precomputed eigenvalues
    k : int, default=5
        number of eigenvalues for KF and Schatten methods
    from_evalues : boolean, default=False
        If True, then L is assumed to be the precomputed eigenvalues
    from_gram : boolean, default=False
        If True, then L is assumed to be a square kernel (Gram) matrix.
        Otherwise an svd will be performed on L where the Gram matrix is LL^T
        which improves computational efficiency.
    Returns
    -------
    complexity_dict : dict
        dictionary of (metric_name, metric_value) pairs for L
    """

    complexity_dict = {}

    # For efficiency across the multiple metrics
    if from_evalues:
        evalues = L
    elif from_gram:
        evalues = np.linalg.svd(L, compute_uv=False, hermitian=True)
    else:
        ss = np.linalg.svd(L, compute_uv=False)
        evalues = np.zeros(L.shape[0])
        evalues[:len(ss)] = ss**2

    KF_norms, KF_ratios, KF_kers, Schattens = get_KF_Schatten_norms(evalues, k, from_evalues=True)
    complexity_dict['KF-raw'] = KF_norms
    complexity_dict['KF-ratio'] = KF_ratios
    complexity_dict['KF-kernel'] = KF_kers
    complexity_dict['Schatten'] = Schattens

    h_star, h_argmin = get_local_rad_bound(evalues, normalize=True, from_evalues=True)
    complexity_dict['h*'] = h_star
    complexity_dict['h_argmin'] = h_argmin

    return complexity_dict


def compute_tau(gen_gap, metric, inverse=False):
    """
    Input: array (generalization gap); array (metric computed for the model instance at such a generalization gap);
    - If inverse: first take inverse of the metric, and compute the kendall tau coefficient
    Return: kendall's tau coefficient, pvalue
    """
    if inverse:
        metric = np.array([1 / elem for elem in metric])
    tau, p_value = stats.kendalltau(gen_gap, metric)
    return tau, p_value


def get_df_tau(plot_dict, gen_err):
    """
    Return a dataframe of the kendall tau's coefficient for different methods
    """
    # tau, p_value = compute_tau(result_dict[err], plot_dict['avg_clusters'], inverse=True)
    # taus, pvalues, names, inverses = [tau], [p_value], ['cc'], ['True']
    taus, pvalues, names, inverses = [], [], [], []
    for key, value in plot_dict.items():
        value = np.array(value)
        # if key in ['ranks', 'stable_ranks', 'avg_clusters', 'modularity']:
        #   continue
        for i in range(value.shape[1]):
            if key == "Schatten":
                if i == 0:  # Schatten 1-norm, no inversion
                    inverse_flag = False
                elif i == 1:
                    continue  # skip trivial 2-norm
                else:
                    inverse_flag = True
            else:
                inverse_flag = True
            tau, p_value = compute_tau(gen_err, value[:, i], inverse=inverse_flag)
            taus.append(tau)
            pvalues.append(p_value)
            names.append(key + "_" + str(i + 1))
            inverses.append(inverse_flag)

    kendal_cor = pd.DataFrame(
        {"metric": names, "kendall_tau": taus, "pvalue": pvalues, "inverse": inverses}
    )

    return kendal_cor