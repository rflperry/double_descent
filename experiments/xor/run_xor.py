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


"""
Experiment Settings
"""

DATA_PARAMS_DICT = {
    "n_train_samples": [4096],
}
N_TEST_SAMPLES = 10000

TREE_PARAMS = {
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_depth": [2, 4, 8, 12, 16, None],
}

FOREST_PARAMS = {
    "n_estimators": [1, 2, 3, 4, 5, 10, 20],
    "bootstrap": [False],
    "max_depth": [2, 3, 4, 6, 8, 10, 15, 20, None],
    "n_jobs": [-1],
}

NETWORK_PARAMETERS = {"temp": [None]}

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
    irm_evals = get_tree_evals(tree, X_train)
    extra_outputs = {"n_total_leaves": tree.get_n_leaves()}
    # same as activation evals

    return y_train_pred, y_test_pred, irm_evals, irm_evals, extra_outputs


run_tree.extra_headers = ["n_total_leaves"]


def run_forest(X_train, y_train, X_test, model_params):
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    irm_evals, act_evals = get_forest_evals(model, X_train)
    extra_outputs = {
        "n_total_leaves": np.sum([tree.get_n_leaves() for tree in model.estimators_])
    }

    return y_train_pred, y_test_pred, irm_evals, act_evals, extra_outputs


run_forest.extra_headers = ["n_total_leaves"]


def run_network(X_train, y_train, X_test, model_params):
    return


"""
Experiment result metrics
"""


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
            "irm_l2",
            "activated_regions",
            "regions_l2",
        ]
        + run_model.extra_headers
    )
    f = open(
        Path(args.output_dir) / f"xor_{args.model}_results.csv",
        "a+" if args.append else "w+",  # optional append
    )
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
                y_train_pred, y_test_pred, irm_evals, act_evals, extras = run_model(
                    X_train, y_train, X_test, model_params
                )

                # Compute metrics
                results = (
                    [args.model, rep]
                    + list(data_params.values())
                    + list(model_params.values())
                    + get_y_metrics(y_train, y_train_pred)  # Train
                    + get_y_metrics(y_test, y_test_pred)  # Test
                    + get_eigenval_metrics(irm_evals, act_evals)
                    + list(extras.values())
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
