"""Runs the Gaussian XOR experiment"""

from argparse import ArgumentParser
from pathlib import Path
import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss, mean_squared_error
from tqdm import tqdm

from partition_decode.dataset import generate_gaussian_parity
from partition_decode.df_utils import get_tree_evals, get_forest_evals, get_forest_irm, get_tree_irm
from partition_decode.models import ReluNetClassifier
from partition_decode.metrics import irm2activations, score_matrix_representation, fast_evals


"""
Experiment Settings
"""

DATA_PARAMS_DICT = {
    "n_train_samples": [500], # [4096],
}
N_TEST_SAMPLES = 10000

TREE_PARAMS = {
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_depth": [2, 4, 8, 12, 16, None],
}

FOREST_PARAMS = {
    "n_estimators": [1, 2, 3, 4, 5, 10, 20],
    # "max_features": [1],
    # "splitter": ['random'],
    "bootstrap": [False],
    "max_depth": [2, 3, 4, 6, 8, 10, 15, 20, None],
    "n_jobs": [-1],
}

NETWORK_PARAMS = {
    'hidden_layer_dims': [
        # [2],
        # [10],
        # [100],
        # [1000],
        [10000],
    ],
    # 'hidden_layer_dims': sum([
    #     [
    #         [2**width_factor]*(1.5**depth_factor).astype(int)
    #         for depth_factor in np.arange(1, 9-width_factor+1)
    #     ] for width_factor in np.arange(1, 9)
    # ], []),
    'n_epochs': [10000],# [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],# 2048],
    'learning_rate': [0.01],
    'batch_size': [128],
    'verbose': [0],
    'learning_rate': [0.01]
}

MODEL_METRICS = {
    'ALL': [
        'IRM_L1', 'IRM_L2', 'n_regions', 'ACTS_L2', 'IRM_h*', 'ACTS_h*', 'entropy',
        'rows_mean_L1', 'rows_mean_L2', 'cols_mean_L1', 'cols_mean_L2'
        ],
    'tree': ['n_leaves'],
    'forest': ['n_total_leaves'],
    'network': [
        # 'irm_l2_pen', 'activated_regions_pen', 'regions_l2_pen',
        'n_parameters', 'depth', 'width'
        ],
}

"""
Experiment run functions
"""


def load_Xy_data(n_train_samples, random_state):
    X, y = generate_gaussian_parity(
        n_samples=n_train_samples, angle_params=0, random_state=random_state
    )
    return X, y


def run_tree(X_train, y_train, X_test, model_params):
    tree = DecisionTreeClassifier(**model_params)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict_proba(X_train)
    y_test_pred = tree.predict_proba(X_test)
    # same as activation evals
    irm = get_tree_irm(tree, X_train)
    model_metrics = get_eigenval_metrics(irm)
    model_metrics += [tree.get_n_leaves()]

    return y_train_pred, y_test_pred, model_metrics


def run_forest(X_train, y_train, X_test, model_params):
    model_params = model_params.copy()
    if "splitter" in model_params.keys():
        splitter = model_params['splitter']
        model_params.pop('splitter')
        model = RandomForestClassifier(**model_params)
        model.base_estimator.splitter = splitter
    else:
        model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict_proba(X_train)
    y_test_pred = model.predict_proba(X_test)
    
    irm = get_forest_irm(model, X_train)
    model_metrics = get_eigenval_metrics(irm)
    model_metrics += [np.sum([tree.get_n_leaves() for tree in model.estimators_])]

    return y_train_pred, y_test_pred, model_metrics


def run_network(X_train, y_train, X_test, model_params):
    model = ReluNetClassifier(**model_params)

    model.fit(X_train, y_train)
    y_train_pred = model.predict_proba(X_train)
    y_test_pred = model.predict_proba(X_test)

    irm = model.get_internal_representation(X_train, penultimate=False)

    model_metrics = get_eigenval_metrics(irm)
    model_metrics += [
        model.n_parameters_, len(model.hidden_layer_dims),
        model.hidden_layer_dims[0]
        ]

    return y_train_pred, y_test_pred, model_metrics


"""
Experiment result metrics
"""


def clean_results(results):
    results = list(results)
    cleaned = []
    for result in results:
        if type(result) == list:
            result = ';'.join(map(str, result))
        cleaned.append(result)
    return cleaned


def get_y_metrics(y_true, y_pred):
    # error = zero_one_loss(y_true, y_pred.argmax(0))
    error = mean_squared_error(y_true, y_pred[:, 1])
    return [error]


def get_eigenval_metrics(irm):
    metric_params = [
        {'metric': 'norm', 'p': 1},
        {'metric': 'norm', 'p': 2},
        {'metric': 'n_regions'},
        {'metric': 'norm', 'p': 2, 'regions': True},
        {'metric': 'h*'},
        {'metric': 'h*', 'regions': True},
        {'metric': 'entropy'},
        {'metric': 'row_means', 'p': 1},
        {'metric': 'row_means', 'p': 2},
        {'metric': 'col_means', 'p': 1},
        {'metric': 'col_means', 'p': 2},
    ]
    metrics = []
    evals = fast_evals(irm)
    for params in metric_params:
        if params['metric'] in ['norm', 'h*', 'entropy'] and (
            'regions' not in list(params.keys())
        ):
            metrics.append(
                score_matrix_representation(evals, is_evals=True, **params)
            )
        else:
            metrics.append(
                score_matrix_representation(irm, **params)
            )
    return metrics


"""
Run Experiment
"""


def main(args):
    # Load invariant test data
    X_test, y_test = load_Xy_data(N_TEST_SAMPLES, random_state=12345)

    # Define model
    if args.model == "tree":
        model_params_dict = TREE_PARAMS
        run_model = run_tree
    elif args.model == "forest":
        model_params_dict = FOREST_PARAMS
        run_model = run_forest
    elif args.model == "network":
        model_params_dict = NETWORK_PARAMS
        run_model = run_network

    header = (
        [
            "model",
            "rep",
        ]
        + list(DATA_PARAMS_DICT.keys())
        + list(model_params_dict.keys())
        + [
            "train_mse",
            "test_mse",
            # "train_01_error",
            # "test_01_error",
        ]
        + MODEL_METRICS['ALL']
        + MODEL_METRICS[args.model]
    )
    f = open(
        Path(args.output_dir) / f"xor_{args.model}_results.csv",
        "a+" if args.append else "w+",  # optional append
    )
    if not args.append:
        f.write(",".join(header) + "\n")
        f.flush()

    # Create combinatiosn of data parameters
    keys, values = zip(*DATA_PARAMS_DICT.items())
    data_params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create combinations of model parameters
    keys, values = zip(*model_params_dict.items())
    model_params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for rep in tqdm(range(args.n_reps)):
        # Repitions

        for data_params in data_params_grid:
            # Create data to test
            X_train, y_train = load_Xy_data(random_state=0 if args.fix_train_data else rep, **data_params)

            for model_params in model_params_grid:
                # Train and test model
                y_train_pred, y_test_pred, model_metrics = run_model(
                    X_train, y_train, X_test, model_params
                )

                # Compute metrics
                results = (
                    [args.model, rep]
                    + list(data_params.values())
                    + clean_results(model_params.values())
                    + get_y_metrics(y_train, y_train_pred)  # Train
                    + get_y_metrics(y_test, y_test_pred)  # Test
                    + model_metrics
                )

                # Dynamically write results
                f.write(",".join(map(str, results)) + "\n")
                f.flush()

    print("Completed")


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs the Gaussian XOR experiment")

    parser.add_argument(
        "--model", choices=["tree", "forest", "network"], help="Experiment to run"
    )
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--append", action="store_true", help="If true, appends to output csv"
    )
    parser.add_argument('--fix_train_data', action='store_true', help='If True, uses the same training seed across all reps')

    args = parser.parse_args()

    main(args)
