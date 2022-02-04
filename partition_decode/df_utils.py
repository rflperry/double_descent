import numpy as np

from joblib import Parallel, delayed
import multiprocessing

from .dataset import generate_gaussian_parity
from .forest import train_forest, get_tree
from .metrics import compute_true_posterior
import os


def run_df_experiment(
    train_n_samples=4096,
    test_n_samples=1000,
    n_reps=100,
    max_node=None,
    n_est=10,
    exp_alias="deep",
):

    grid_xx, grid_yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    grid_true_posterior = np.array(
        [compute_true_posterior(x) for x in (np.c_[grid_xx.ravel(), grid_yy.ravel()])]
    )

    train_mean_error, test_mean_error = [], []
    train_mean_error_log, test_mean_error_log = [], []
    gini_train_mean_score, gini_test_mean_score = [], []

    X_train, y_train = generate_gaussian_parity(
        n_samples=train_n_samples, angle_params=0
    )
    # X_test, y_test = generate_gaussian_parity(n_samples=test_n_samples, angle_params=0)

    method = "rf"

    if max_node is None:
        rf = get_tree(method, max_depth=None)
        rf.fit(X_train, y_train)
        if method == "gb":
            max_node = (
                sum([estimator[0].get_n_leaves() for estimator in rf.estimators_])
            ) + 50
        else:
            max_node = (
                sum([estimator.get_n_leaves() for estimator in rf.estimators_]) + 50
            )

    train_error, test_error = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    hellinger_dist = [list() for _ in range(n_reps)]
    nodes = [list() for _ in range(n_reps)]
    polys = [list() for _ in range(n_reps)]
    # for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):
    # for rep_i in tqdm(range(n_reps), position=0, leave=True):
    def one_run(rep_i):
        [
            nodes[rep_i],
            polys[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            hellinger_dist[rep_i],
        ] = train_forest(
            method,
            grid_xx,
            grid_yy,
            grid_true_posterior,
            rep_i,
            train_n_samples=train_n_samples,
            test_n_samples=test_n_samples,
            max_node=max_node,
            n_est=n_est,
            exp_name=exp_alias,
        )

    # n_cores = multiprocessing.cpu_count()
    # Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(n_reps))
    # Forests are already parallelized
    for i in range(n_reps):
        one_run(i)

    train_mean_error = np.array(train_error).mean(axis=0)
    test_mean_error = np.array(test_error).mean(axis=0)
    train_mean_error_log = np.array(train_error_log).mean(axis=0)
    test_mean_error_log = np.array(test_error_log).mean(axis=0)
    nodes_mean = np.array(nodes).mean(axis=0)
    gini_train_mean_score = np.array(gini_score_train).mean(axis=0)
    gini_test_mean_score = np.array(gini_score_test).mean(axis=0)

    error_dict = {
        "train_err": train_mean_error,
        "test_err": test_mean_error,
        "train_err_log": train_mean_error_log,
        "test_err_log": test_mean_error_log,
        "train_gini": gini_train_mean_score,
        "test_gini": gini_test_mean_score,
        "nodes": nodes_mean,
    }
    return error_dict


def read_df_results(n_reps, exp_alias="depth"):

    dir_path = "../results/df/" + exp_alias + "/"
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                file_paths.append(os.path.join(root, file))

    result = lambda: None
    train_error, test_error = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    nodes = [list() for _ in range(n_reps)]
    polytopes = [list() for _ in range(n_reps)]
    hellinger_dist = [list() for _ in range(n_reps)]

    for rep_i in range(n_reps):
        [
            nodes[rep_i],
            polytopes[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            hellinger_dist[rep_i],
        ] = np.load(
            file_paths[rep_i],
        )
        # np.load("../results/xor_rf_dd_" + exp_alias + "_" + str(rep_i) + ".npy")

    result.train_err_list = np.array(train_error)  # .mean(axis=0)
    result.test_err_list = np.array(test_error)  # .mean(axis=0)
    result.train_err_log_list = np.array(train_error_log)  # .mean(axis=0)
    result.test_error_log_list = np.array(test_error_log)  # .mean(axis=0)
    result.n_nodes = np.array(nodes).mean(axis=0)
    result.n_polytopes_list = np.array(polytopes)  # .mean(axis=0)
    result.gini_train_list = np.array(gini_score_train)  # .mean(axis=0)
    result.gini_test_list = np.array(gini_score_test)  # .mean(axis=0)
    result.hellinger_dist_list = np.array(hellinger_dist)  # .mean(axis=0)

    return result


"""
Eigenvalues utilities
"""


def get_tree_irm(tree, X, scale=False):
    n = X.shape[0]
    leaf_indices = tree.apply(X)
    irm = np.zeros((n, len(set(leaf_indices))))
    irm[np.arange(n), np.unique(leaf_indices, return_inverse=True)[1]] = 1
    if scale: # weighting by size of leaf, proper weights
        irm /= np.sqrt(irm.sum(0, keepdims=True))

    return irm


def get_forest_irm(forest, X, scale=False):
    tree_irms = [get_tree_irm(tree, X, scale=scale) for tree in forest.estimators_]
    if scale:
        return np.hstack(tree_irms) / np.sqrt(len(tree_irms))
    else:
        return np.hstack(tree_irms)


def get_tree_evals(model, X):
    """
    Returns the eigenvalues of the tree's internal representation,
    the leaf similarity matrix. The eigenvalues are the sizes
    of each of the leaves.

    Returns
    -------
    irm : internal representation eigenvalues
    """
    leaf_indices = model.apply(X)
    _,  evals = np.unique(leaf_indices, return_counts=True)
    return evals


def get_forest_evals(model, X):
    """
    Returns the eigenvalues of the forests's internal representation,
    the average of all its tree's representations, as well as its
    activation matrix.

    Returns
    -------
    irm_evals : internal representation eigenvalues
    act_evals : activation matrix eigenvalues
    """
    n = X.shape[0]
    irm = np.zeros((n, n))
    for tree in model.estimators_:
        leaf_indices = tree.apply(X)
        Z = np.zeros((n, len(set(leaf_indices))))
        Z[np.arange(n), np.unique(leaf_indices, return_inverse=True)[1]] = 1
        irm += Z @ Z.T
    irm /= len(model.estimators_)
    irm_evals = np.linalg.svd(irm, compute_uv=False, hermitian=True)
    irm_evals[np.abs(irm_evals) < 1e-10] = 0

    act_mat = (irm == 1.0).astype(int)
    act_evals = np.linalg.svd(act_mat, compute_uv=False, hermitian=True)
    act_evals[np.abs(act_evals) < 1e-10] = 0

    return irm_evals, act_evals


def get_tree_weights(tree, X):
    """
    Computes the leaf weights for a regression tree.

    Parameters
    ----------
    tree : sklearn.tree.DecisionTreeRegressor
        Fitted tree with l leaves and k targets
    X : np.ndarray, shape (n, d)
        Training data
    
    Returns
    -------
    weights : np.ndarray, shape (k, l)
    """
    n = X.shape[0]
    leaves = tree.apply(X).reshape(n, -1) # n, 1 w/ L unique
    y_hat = tree.predict(X) # n, k
    _, indices = np.unique(leaves, axis=0, return_index=True)
    weights = np.asarray([
        y_hat[indices, i]
        for i in range(y_hat.shape[1])
    ])
    return weights
    

def get_forest_weights(model, X, expanded=True):
    """
    Computes the leaf weights for a regression tree. These
    are the tree weights normalized by the number of trees.

    Parameters
    ----------
    tree : sklearn.ensemble.RandomForestRegressor
        Fitted tree with l total leaves, k targets, and t trees
    X : np.ndarray, shape (n, d)
        Training data
    expanded : bool, default=False
        If True, returns a 3D block diagonal tensor with
        weights from each tree separated along the final axis.
    Returns
    -------
    weights : np.ndarray
        If expanded=True, shape (k, l, t). Else shape (k, l)
    
    Note
    ----
    expanded=False is not informative at max depth as each sample
    receives the same overfitted prediction. Thus the weights
    are constant.
    """
    tree_weights = [
        get_tree_weights(tree, X)
        for tree in model.estimators_
    ]
    
    if expanded:
        shapes = [i.shape for i in tree_weights]
        out = np.zeros([shapes[0][0], sum(s[1] for s in shapes), len(shapes)])
        
        r = 0
        for i, (_, rr) in enumerate(shapes):
            out[:, r:r + rr, i] = tree_weights[i]
            r += rr
    else:
        out = np.concatenate(tree_weights, axis=-1)
        out = out.reshape(*out.shape, 1)
    
    return out / len(tree_weights)


"""
  Example to run the `Deep RF` vs `Shallow RF` experiments
  and plot the figure.
"""
### via CLI

# import argparse

# parser = argparse.ArgumentParser(description='Run a double descent experiment.')

# parser.add_argument('--deep', action="store_true", default=False)
# parser.add_argument('-n_reps', action="store", dest="n_reps", type=int)
# parser.add_argument('-n_est', action="store", dest="n_est", type=int)
# parser.add_argument('-max_node', action="store", dest="max_node", default=None, type=int)
# parser.add_argument('-cov_scale', action="store", dest="cov_scale", default=1.0, type=float)


# args = parser.parse_args()

# exp_alias = "deep" if args.deep else "shallow"

# result = rf_dd_exp(max_node=args.max_node, n_est=args.n_est, n_reps=args.n_reps, exp_alias = exp_alias)

# --------------

### on a notebook
# n_reps = 1  # the number of repetitions of a single run of the algorithm
# # Run DeepRF
# error_deep = run_df_experiment(max_node=None, n_est=10, n_reps=n_reps, exp_alias="deep")
# # Run ShallowRF
# error_shallow = run_df_experiment(max_node=15, n_est=100, n_reps=n_reps, exp_alias="shallow")
# # np.save('errors.npy', [error_5, error_dd])

# error_deep = read_df_results(n_reps, exp_alias="deep")
# error_shallow = read_df_results(n_reps, exp_alias="shallow")

# results = [error_deep, error_shallow]
# titles = ["RF with overfitting trees", "RF with shallow trees"]

# from .plots import plot_df_results
# plot_df_results(results, titles)
