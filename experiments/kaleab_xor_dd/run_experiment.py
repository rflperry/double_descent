"""Runs the Gaussian XOR experiment"""

from argparse import ArgumentParser
from pathlib import Path
import itertools
import torch

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import zero_one_loss, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from joblib import Parallel, delayed
import os

from partition_decode.dataset import (
    generate_gaussian_parity,
    recursive_gaussian_parity,
    generate_spirals,
    load_mnist,
    samples_trunk,
)
from partition_decode.df_utils import (
    get_tree_evals,
    get_forest_evals,
    get_forest_irm,
    get_tree_irm,
    get_forest_weights,
)
from partition_decode.models import ReluNetClassifier, ReluNetRegressor, KaleabNet
from partition_decode.metrics import (
    irm2activations,
    score_matrix_representation,
    fast_evals,
    mse_classification,
)
from partition_decode.dn_utils import get_norm_irm


"""
Experiment Settings
"""

N_TRAIN_SAMPLES = 1000
N_TEST_SAMPLES = 8192

DATA_PARAMS_DICT = {
    "xor": {
        "n_train_samples": [N_TRAIN_SAMPLES],  # [4096],
        "n_test_samples": [N_TEST_SAMPLES],
        "recurse_level": [0],
        "cov_scale": [1],
        "onehot": [True],
        "noise_dims": [0],
        "shuffle_label_frac": [None]
    },
    "xor90" : {
        "n_train_samples": [N_TRAIN_SAMPLES],  # [4096],
        "n_test_samples": [N_TEST_SAMPLES],
        "recurse_level": [0],
        "angle_params" : [45],
        "cov_scale": [0.15],
        "onehot": [True],
        "noise_dims": [0],
        "shuffle_label_frac": [None]
    },
    "trunk" : {
        "n_train_samples": [100], 
        "n_test_samples": [N_TEST_SAMPLES],
        "dims": [1000],
        "onehot": [True],
        "shuffle_label_frac": [None],
    },
    "spiral": {
        "n_train_samples": N_TRAIN_SAMPLES,  # [4096],
        "n_test_samples": N_TEST_SAMPLES,
         "onehot": [True],
    },
    "mnist": {
        "n_train_samples": [4000],
        "n_test_samples": [10000],
        "save_path": ["/mnt/ssd3/ronan/pytorch"],
        "onehot": [True],
        "shuffle_label_frac": [None], # np.linspace(0, 1, 11), # 
    },
}

DEPTH_FOREST_PARAMS = {
    "n_estimators": [3], # [1, 2, 3, 4, 5, 7, 10, 13, 16, 20],
    "max_features": [1],
    # "splitter": ['random'],
    "bootstrap": [False],
    "max_depth": list(range(1, 25)), # 
    "n_jobs": [-2],
}

SHALLOW_FOREST_PARAMS = {
    "n_estimators": [1, 2, 3, 4, 5, 7, 10, 13, 16, 20, 40],
    "max_features": [1],
    # "splitter": ['random'],
    "bootstrap": [False],
    "max_depth": [5], # list(range(1, 25)), # 
    "n_jobs": [-2],
}

WIDTH_NETWORK_PARAMS = {
    # "hidden_layer_dims": [
    #     [i]*3
    #     for i in range(1, 71)
    # ],
    "hidden_layer_dims": [
        [i]*3
        for i in [1, 2, 3, 4, 6, 8, 16, 20, 24, 32, 48, 64, 96, 128, 256]#[4, 8, 10, 12, 14, 16, 24, 32, 48, 64, 128]
    ],
    "n_epochs": [1000],
    "learning_rate": [1e-2],
    # "batch_size": [32],
    "verbose": [0],
    "early_stop_thresh": [None],
    "bias": [True],
    # "init_prior_model": [False],
}

DEPTH_NETWORK_PARAMS = {
    # "hidden_layer_dims": [
    #     [i]*i if i < 5 else [20]*i
    #     for i in range(1, 20)
    # ],
    "hidden_layer_dims": [
        [20]*i
        for i in [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30]
    ],
    "n_epochs": [1000],
    "learning_rate": [1e-2],
    # "batch_size": [32],
    "verbose": [0],
    "early_stop_thresh": [None],
    "bias": [True],
    # "init_prior_model": [False],
}

NETWORK_PARAMS = {
    "hidden_layer_dims": [
        [width]*depth
        for width, depth in itertools.product(
            [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
            [1, 2, 3, 4, 6, 8, 10, 12, 15, 20],
        )
    ],
    "n_epochs": [1000],
    "learning_rate": [1e-2],
    # "batch_size": [32],
    "verbose": [0],
    "early_stop_thresh": [None],
    "bias": [True],
    # "init_prior_model": [False],
}

MODEL_METRICS = {
    "ALL": [
        "IRM_L0",
        "IRM_L1",
        "IRM_L2",
        "n_regions",
        "ACTS_L2",
        "IRM_entropy",
    ],
    "deep_forest": [
        "n_total_leaves",
        ],
    "shallow_forest": [
        "n_total_leaves",
        ],
}

for key in ['wide_relu', 'deep_relu', 'relu']:
    MODEL_METRICS[key] = [
        "PEN_IRM_L0",
        "PEN_IRM_L1",
        "PEN_IRM_L2",
        "PEN_n_regions",
        "PEN_ACTS_L2",
        "PEN_IRM_entropy",
        "n_parameters",
        "depth",
        "width",
        "kernel_trace",
        "head_norm",
    ]

"""
Experiment run functions
"""


def load_Xy_data(dataset, n_samples, random_state, data_params, train=None, onehot=False, shuffle_label_frac=None):
    if dataset == "xor":
        X, y = generate_gaussian_parity(
            n_samples=n_samples,
            # recurse_level=data_params["recurse_level"],
            angle_params=0,
            random_state=random_state,
            cov_scale=data_params["cov_scale"],
            noise_dims=data_params["noise_dims"],
        )
    elif dataset == "spiral":
        X, y = generate_spirals(n_samples=n_samples, random_state=random_state)
    elif dataset == "mnist":
        X, y = load_mnist(
            n_samples=n_samples, save_path=data_params["save_path"], train=train
        )
    elif dataset == "trunk":
        X, y = samples_trunk(n_samples = n_samples, dims=data_params["dims"], random_state=random_state)
    elif dataset == "xor90":
        X, y = generate_gaussian_parity(
            n_samples=n_samples,
            # recurse_level=data_params["recurse_level"],
            angle_params=data_params["angle_params"],
            random_state=random_state,
            cov_scale=data_params["cov_scale"],
            noise_dims=data_params["noise_dims"],
        )

    if onehot and y.ndim == 1:
        new_y = np.zeros((y.shape[0], len(np.unique(y))))
        new_y[np.arange(new_y.shape[0]), y] = 1
        y = new_y

    if shuffle_label_frac is not None:
        # np.random.seed(random_state)
        idx = np.random.choice(y.shape[0], int(y.shape[0] * shuffle_label_frac),replace=False)
        old_vals = y[idx]
        np.random.shuffle(old_vals)
        y[idx] = old_vals

    return X, y


def get_posterior_map(clf, regressor=False):
    xx, yy = np.meshgrid(
        np.arange(-3, 3, 0.1),
        np.arange(-3, 3, 0.1))
    if regressor:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if Z.ndim > 1 and Z.shape[1] > 1:
        Z = Z[:, 0]
    Z = Z.reshape(xx.shape)
    return Z


def run_forest(X_train, y_train, X_test, model_params, model=None, save_path=None):
    from sklearn.ensemble import RandomForestRegressor
    model_params = model_params.copy()
    if "splitter" in model_params.keys():
        model = RandomForestRegressor(**del_dict_keys(model_params, ["splitter"]))
        model.base_estimator.splitter = model_params["splitter"]
    else:
        model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    Z = get_posterior_map(model, regressor=True)

    irm = get_forest_irm(model, X_train)

    model_metrics = get_eigenval_metrics(irm, eval_divisor=model.n_estimators)
    model_metrics += [np.sum([tree.get_n_leaves() for tree in model.estimators_])]

    return model, y_train_pred, y_test_pred, model_metrics, Z


def run_relu_classifier(X_train, y_train, X_test, model_params, prior_model=None, save_path=None):

    # if prior_model is not None and (
    #     "init_prior_model" in model_params.keys() and model_params["init_prior_model"]
    # ):
    #     model = ReluNetRegressor(**del_dict_keys(model_params, ["init_prior_model"]))
    #     model._build_model(X_train.shape[-1], y_train.shape[1])

    #     # Stability of decreasing training error
    #     if model.n_parameters_ < X_train.shape[0] * y_train.shape[1]:
    #         with torch.no_grad():
    #             for prior_layer, new_layer in zip(prior_model.model_, model.model_,):
    #                 if isinstance(new_layer, torch.nn.ReLU):
    #                     continue
    #                 width, depth = prior_layer.weight.shape
    #                 new_layer.weight[:width, :depth] = prior_layer.weight
    #                 new_layer.bias[:width] = prior_layer.bias

    # else:
    #     model = ReluNetRegressor(**del_dict_keys(model_params, ["init_prior_model"]))
    #     model._build_model(X_train.shape[1], y_train.shape[1])

    model = KaleabNet(**model_params)
    model._build_model(X_train.shape[1], y_train.shape[1])
    model._train_model(X_train, y_train)

    y_train_pred = model.predict_proba(X_train)
    y_test_pred = model.predict_proba(X_test)

    Z = get_posterior_map(model, regressor=False)

    irm = model.get_internal_representation(X_train, penultimate=False)
    pen_irm = model.get_internal_representation(X_train, penultimate=True, binary=False)

    model_metrics = get_eigenval_metrics(irm, irm.shape[1])
    model_metrics += get_eigenval_metrics(
        (pen_irm > 0).astype(int),
        pen_irm.shape[1])

    model_metrics += [
        model.n_parameters_,
        len(model.hidden_layer_dims),
        model.hidden_layer_dims[0],
        np.sum(np.linalg.svd(pen_irm, compute_uv=False)**2 / pen_irm.shape[1]),
        np.mean(np.linalg.norm(model.model_[-1].weight.detach().numpy(), axis=1)),
    ]

    model_metrics = list(np.round(model_metrics, np.ceil(np.log10(y_train.shape[0])).astype(int)))

    if save_path is not None:
        torch.save(model.model_.state_dict(), save_path)

    return model, y_train_pred, y_test_pred, model_metrics, Z


"""
Experiment result metrics
"""

def del_dict_keys(d, keys):
    new_d = d.copy()
    for key in keys:
        new_d.pop(key)
    return new_d


def clean_results(results):
    results = list(results)
    cleaned = []
    for result in results:
        if type(result) == list:
            result = ";".join(map(str, result))
        cleaned.append(result)
    return cleaned


def get_y_metrics(y_true, y_pred):
    if y_true.ndim == 1:
        errors = [
            zero_one_loss(y_true, y_pred.argmax(1)),
            mse_classification(y_true, y_pred),
        ]
    else:
        errors = [
            zero_one_loss(y_true.argmax(1), y_pred.argmax(1)),
            mean_squared_error(y_true, y_pred),
        ]
    return list(np.round(errors, np.ceil(np.log10(y_true.shape[0])).astype(int)))


def get_eigenval_metrics(irm, eval_divisor=1):
    metric_params = [
        {"metric": "norm", "p": 0},
        {"metric": "norm", "p": 1},
        {"metric": "norm", "p": 2},
        {"metric": "n_regions"},
        {"metric": "norm", "p": 2, "regions": True},
        {"metric": "entropy"},
    ]

    metrics = []
    evals = fast_evals(irm)
    evals /= eval_divisor
    for params in metric_params:
        if params["metric"] in ["norm", "h*", "entropy"] and (
            "regions" not in list(params.keys())
        ):
            metrics.append(score_matrix_representation(evals, is_evals=True, **params))
        elif params["metric"] == 'mean_dot_product':
            metrics.append(score_matrix_representation(irm / eval_divisor, **params))
        else:
            metrics.append(score_matrix_representation(irm, **params))
    return metrics


"""
Run Experiment
"""


def main(args):
    # Define model
    if args.model == "tree":
        model_params_dict = TREE_PARAMS
        run_model = run_tree
    elif args.model == "shallow_forest":
        model_params_dict = SHALLOW_FOREST_PARAMS
        run_model = run_forest
    elif args.model == "deep_forest":
        model_params_dict = DEPTH_FOREST_PARAMS
        run_model = run_forest
    elif args.model == "knn":
        model_params_dict = KNN_PARAMS
        run_model = run_knn
    elif args.model == "rrf":
        model_params_dict = RRF_PARAMS
        run_model = run_rrf
    elif args.model == "relu":
        model_params_dict = NETWORK_PARAMS
        run_model = run_relu_classifier
    elif args.model == "deep_relu":
        model_params_dict = DEPTH_NETWORK_PARAMS
        run_model = run_relu_classifier
    elif args.model == "wide_relu":
        model_params_dict = WIDTH_NETWORK_PARAMS
        run_model = run_relu_classifier

    # Create combinatiosn of data parameters
    keys, values = zip(*DATA_PARAMS_DICT[args.dataset].items())
    data_params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create combinations of model parameters
    keys, values = zip(*model_params_dict.items())
    model_params_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create folder to save results to
    if args.output_dir is None:
        if len(data_params_grid) > 1:
            args.output_dir = f"varying_data_results/{args.dataset}"
        else:
            args.output_dir = f"varying_model_results/{args.dataset}"

    # Create results csv header
    header = (
        ["model", "rep",]
        + list(DATA_PARAMS_DICT[args.dataset].keys())
        + list(model_params_dict.keys())
        + [f"train_01_error", f"train_mse", f"test_01_error", f"test_mse",]
        + MODEL_METRICS["ALL"]
        + MODEL_METRICS[args.model]
    )
    f = open(
        Path(args.output_dir) / f"{args.dataset}_{args.model}_results.csv",
        "a+" if args.append else "w+",  # optional append
    )
    if not args.append:
        f.write(",".join(header) + "\n")
        f.flush()
        save_idx = 0
    else:
        save_idx = sum(1 for _ in open(Path(args.output_dir) / f"{args.dataset}_{args.model}_results.csv", 'r')) - 1


    for data_params in data_params_grid:
        print('Data params: ' + str(data_params))
        # Load invariant test data
        X_test, y_test = load_Xy_data(
            dataset=args.dataset,
            n_samples=data_params["n_test_samples"],
            random_state=12345,
            data_params=data_params,
            train=False,
            onehot=data_params['onehot'],
            shuffle_label_frac=data_params['shuffle_label_frac'],
        )

        # Iterate over repetitions
        for rep in tqdm(range(args.n_reps)):
            # for data_params in data_params_grid:
            # Create data to test
            if rep == 0 or not args.fix_train_data:
                X_train, y_train = load_Xy_data(
                    dataset=args.dataset,
                    n_samples=data_params["n_train_samples"],
                    random_state=0 if args.fix_train_data else rep,
                    data_params=data_params,
                    train=True,
                    onehot=data_params['onehot'],
                    shuffle_label_frac=data_params['shuffle_label_frac'],
                )
                # min_max_scaler = MinMaxScaler()
                # X_train = min_max_scaler.fit_transform(X_train)

            model = None

            # Iterate over models
            for model_params in model_params_grid:
                if args.model == 'forest' and model_params['n_estimators'] > 1 and model_params['max_depth'] != None:
                    continue

                save_path = None
                if args.save_models:
                    save_path = Path(args.output_dir) / "models" / f"{args.dataset}_{args.model}_model_{save_idx}.pkl"

                # Train and test model
                model, y_train_pred, y_test_pred, model_metrics, posteriors = run_model(
                    X_train, y_train, X_test, model_params, model, save_path
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

                # Dynamically write results to csv
                f.write(",".join(map(str, results)) + "\n")
                f.flush()

                directory = Path(args.output_dir) / f"{args.dataset}_{args.model}_posteriors"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                np.save(directory / f"posterior_map_{save_idx}", posteriors)
                save_idx += 1

    print("Completed")


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs experiment")

    parser.add_argument(
        "--model", choices=["tree", "deep_forest", "shallow_forest", "knn", "rrf", "wide_relu", "deep_relu", 'relu'], help="Experiment to run"
    )
    parser.add_argument(
        "--dataset", choices=["xor", "spiral", "mnist", "trunk", "xor90"], help="Experiment to run"
    )
    parser.add_argument("--n_reps", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--append", action="store_true", help="If true, appends to output csv"
    )
    # parser.add_argument(
    #     "--metric", choices=["mse", "01_error"], help="scoring metric to use", default='mse'
    # )
    parser.add_argument(
        "--fix_train_data",
        action="store_true",
        help="If True, uses the same training seed across all reps",
    )
    parser.add_argument(
        "--save_models", action="store_true", help="If true, saves models"
    )

    args = parser.parse_args()

    main(args)
