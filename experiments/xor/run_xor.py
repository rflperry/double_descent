"""Runs the Gaussian XOR experiment"""

from argparse import ArgumentParser
from pathlib import Path
import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from tqdm import tqdm

from partition_decode.dataset import generate_gaussian_parity
from partition_decode.df_utils import get_tree_evals, get_forest_evals
from partition_decode.models import ReluNetClassifier
from partition_decode.dn_utils import irm2activations


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
        [10],
        [10, 10],
        [10, 10, 10],
        # [10, 10, 10, 10],
        [100],
        [100, 100],
        # [100, 100, 100],
        [1000],
        # [1000, 1000],
        [1000, 1000, 1000],
        # [10000],
    ],
    # 'hidden_layer_dims': [
    #     [5],
    #     [5, 5],
    #     [5, 5, 5],
    #     [5, 5, 5, 5],
    #     [5, 5, 5, 5, 5],
    #     [10],
    #     [10, 10],
    #     [10, 10, 10],
    #     [10, 10, 10, 10, 10],
    #     [50, 50],
    #     [50, 50, 50],
    #     [50, 50, 50, 50],
    #     [100],
    #     [100, 100],
    #     [100, 100, 100],
    #     [1000],
    #     [1000, 1000]
    # ],
    # 'hidden_layer_dims': sum([
    #     [
    #         [2**width_factor]*(1.5**depth_factor).astype(int)
    #         for depth_factor in np.arange(1, 9-width_factor+1)
    #     ] for width_factor in np.arange(1, 9)
    # ], []),
    'n_epochs': [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'learning_rate': [0.01],
    'batch_size': [128],
    'verbose': [0],
    'learning_rate': [0.01]
}

MODEL_METRICS = {
    'tree': ['irm_l2', 'activated_regions', 'regions_l2' 'n_leaves'],
    'forest': ['irm_l2', 'activated_regions', 'regions_l2', 'n_total_leaves'],
    'network': [
        'irm_l2_pen', 'activated_regions_pen', 'regions_l2_pen',
        'irm_l2_all', 'activated_regions_all', 'regions_l2_all',
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
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    # same as activation evals
    irm_evals = get_tree_evals(tree, X_train)
    model_metrics = get_eigenval_metrics(irm_evals, irm_evals)
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
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    irm_evals, act_evals = get_forest_evals(model, X_train)
    model_metrics = get_eigenval_metrics(irm_evals, act_evals)
    model_metrics += [np.sum([tree.get_n_leaves() for tree in model.estimators_])]

    return y_train_pred, y_test_pred, irm_evals, act_evals, model_metrics


def run_network(X_train, y_train, X_test, model_params):
    model = ReluNetClassifier(**model_params)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    irm = model.get_internal_representation(X_train, penultimate=False)
    irm_evals_pen = np.linalg.svd(
        irm[:, -model.hidden_layer_dims[-1]:],
        compute_uv=False)**2 / irm.shape[1]
    irm_evals_all = np.linalg.svd(
        irm, 
        compute_uv=False)**2 / irm.shape[1]

    act_evals_all = np.linalg.svd(
        irm2activations(irm),
        compute_uv=False)**2
    act_evals_pen = np.linalg.svd(
        irm2activations(irm[:, -model.hidden_layer_dims[-1]:]),
        compute_uv=False)**2

    model_metrics = get_eigenval_metrics(irm_evals_pen, act_evals_pen)
    model_metrics += get_eigenval_metrics(irm_evals_all, act_evals_all)
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
    error = zero_one_loss(y_true, y_pred)
    return [error]


def get_eigenval_metrics(irm_evals, act_evals):
    irm_l2 = np.linalg.norm(irm_evals, ord=2)
    activated_regions = np.sum(act_evals > 0)
    regions_l2 = np.linalg.norm(act_evals, ord=2)
    return [irm_l2, activated_regions, regions_l2]


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
            "train_01_error",
            "test_01_error",
        ]
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
            X_train, y_train = load_Xy_data(random_state=rep, **data_params)

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

    args = parser.parse_args()

    main(args)
