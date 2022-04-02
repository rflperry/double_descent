import numpy as np
import torch
from torch import nn

import copy
import random

import os

## Distributions


def generate_gaussian_parity(
    n_samples,
    means=None,
    cov_scale=1,
    angle_params=None,
    random_state=None,
    noise_dims=0,
    noise_std=1,
):
    """
    Generate 2-dimensional Gaussian XOR distribution, a mixture of four Gaussian belonging to two classes.
    (Classic XOR problem but each point is the center of a Gaussian blob distribution)

    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four clusters with equal probability.

    means : ndarray of shape [n_centers,2], default=None
        The coordinates of the center of total n_centers blobs.

    cov_scale : float, default=1
        The standard deviation of the blobs.

    angle_params: float, default=None
        Number of degrees to rotate the distribution by.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if means is None:
        means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]

    if angle_params is None:
        angle_params = np.random.uniform(0, 2 * np.pi)

    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=n_samples // 4
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate(
        [np.ones((n_samples // 4)) * int(i < 2) for i in range(len(means))]
    )
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )

    X_noise = np.random.normal(0, 1, (n_samples, noise_dims))
    X = np.hstack((X, X_noise))

    return X, Y.astype(int)


def recursive_gaussian_parity(
    n_samples,
    means=None,
    cov_scale=1,
    recurse_level=1,
    angle_params=None,
    random_state=None,
    noise_dims=0,
):
    """
    Generate 2-dimensional distribution akin to Gaussian XOR but two adjacent
    partitions of the four XOR partitions are themselves smaller Gaussian XOR.

    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four clusters with equal probability.

    cov_scale : float, default=1
        The standard deviation of the blobs.

    recurse_level : int, default=1
        Number of times to recurse

    angle_params: float, default=None
        Number of radians to rotate the distribution by.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    means = np.asarray([[-1, -1], [1, 1], [1, -1], [-1, 1]])

    X, y = generate_gaussian_parity(
        n_samples,
        means=means,
        cov_scale=cov_scale,
        angle_params=angle_params,
        random_state=random_state,
        noise_dims=noise_dims,
    )
    if recurse_level == 0:
        return X, y

    X_recurs, y_recurs = recursive_gaussian_parity(
        n_samples // 4,
        means=means,
        recurse_level=recurse_level-1,
        cov_scale=cov_scale,
        angle_params=angle_params,
        random_state=random_state,
        noise_dims=noise_dims,
    )

    # Recurse impute
    adj_indices = [
        np.arange(n_samples // 4),
        np.arange(3 * n_samples // 4, n_samples),
    ]

    for idx, mean in zip(adj_indices, means[[0, 3]]):
        X[idx] = X_recurs
        y[idx] = y_recurs
        X[idx] /= 2
        X[idx] += mean
        

    return X, y


def get_dataset(
    n_samples=1000, one_hot=False, cov_scale=1, include_hybrid=False, random_state=None
):
    """
    Generate the Gaussian XOR dataset and move it to gpu

    Parameters
    ----------
    n_samples : int
        Total number of points in the Gaussian XOR dataset.

    one_hot : bool, default=False
        A boolean indicating if the label should one hot encoded.

    cov_scale : float, default=1
        The standard deviation of the blobs.

    include_hybrid: bool, default=False
        A boolean indicating if hybrid set should be included for computing average stability.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    train_x : Tensor [n_samples, 2]
        Training set features

    train_y: Tensor [n_samples]
        Training set labels

    test_x : Tensor [n_samples, 2]
        Test set features

    test_y: Tensor [n_samples]
        Test set labels
    """

    use_gpa = torch.cuda.is_available()
    if use_gpa:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if include_hybrid:
        D_x, D_y = generate_gaussian_parity(
            cov_scale=cov_scale,
            n_samples=(2 * n_samples),
            angle_params=0,
            random_state=random_state,
        )
        D_perm = np.random.permutation(2 * n_samples)
        D_x, D_y = D_x[D_perm, :], D_y[D_perm]
        train_x, train_y = D_x[:n_samples], D_y[:n_samples]
        ghost_x, ghost_y = D_x[n_samples:], D_y[n_samples:]
        hybrid_sets = []
        rand_idx = random.sample(range(0, n_samples - 1), n_samples // 10)
        for rand_i in rand_idx:
            hybrid_x, hybrid_y = np.copy(train_x), np.copy(train_y)
            hybrid_x[rand_i], hybrid_y[rand_i] = ghost_x[rand_i], ghost_y[rand_i]
            hybrid_x = torch.FloatTensor(hybrid_x)
            hybrid_y = torch.FloatTensor(hybrid_y).unsqueeze(-1)
            if use_gpa:
                hybrid_x, hybrid_y = hybrid_x.cuda(), hybrid_y.cuda()
            hybrid_sets.append((hybrid_x, hybrid_y))
    else:
        train_x, train_y = generate_gaussian_parity(
            cov_scale=cov_scale,
            n_samples=n_samples,
            angle_params=0,
            random_state=random_state,
        )
        train_perm = np.random.permutation(n_samples)
        train_x, train_y = train_x[train_perm, :], train_y[train_perm]

    test_x, test_y = generate_gaussian_parity(
        cov_scale=cov_scale,
        n_samples=2 * n_samples,
        angle_params=0,
        random_state=random_state,
    )

    test_perm = np.random.permutation(2 * n_samples)
    test_x, test_y = test_x[test_perm, :], test_y[test_perm]

    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y = torch.FloatTensor(train_y).unsqueeze(-1)  # [:,0]
    test_y = torch.FloatTensor(test_y).unsqueeze(-1)  # [:,0]

    if one_hot:
        train_y = torch.nn.functional.one_hot(train_y[:, 0].to(torch.long))
        test_y = torch.nn.functional.one_hot(test_y[:, 0].to(torch.long))

    # move to gpu
    if use_gpa:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()

    if include_hybrid:
        return train_x, train_y, test_x, test_y, hybrid_sets

    return train_x, train_y, test_x, test_y


def generate_spirals(
    n_samples,
    n_class=2,
    noise=1,
    random_state=None,
):
    """
    Generate 2-dimensional spiral simulation
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the individual spirals.
    n_class : array of shape [n_centers], optional (default=2)
        Number of class for the spiral simulation.
    noise : float, optional (default=0.3)
        Parameter controlling the spread of each class.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    X = []
    y = []

    if n_class == 2:
        turns = 2
    elif n_class == 3:
        turns = 2.5
    elif n_class == 5:
        turns = 3.5
    elif n_class == 7:
        turns = 4.5
    else:
        raise ValueError("sorry, can't currently support %s classes " % n_class)

    mvt = np.random.multinomial(n_samples, 1 / n_class * np.ones(n_class))

    if n_class == 2:
        r = np.random.uniform(0, 1, size=int(n_samples / n_class))
        r = np.sort(r)
        t = np.linspace(
            0, np.pi * 4 * turns / n_class, int(n_samples / n_class)
        ) + np.random.normal(0, noise, int(n_samples / n_class))
        dx = r * np.cos(t)
        dy = r * np.sin(t)

        X.append(np.vstack([dx, dy]).T)
        X.append(np.vstack([-dx, -dy]).T)
        y += [0] * int(n_samples / n_class)
        y += [1] * int(n_samples / n_class)
    else:
        for j in range(1, n_class + 1):
            r = np.linspace(0.01, 1, int(mvt[j - 1]))
            t = (
                np.linspace(
                    (j - 1) * np.pi * 4 * turns / n_class,
                    j * np.pi * 4 * turns / n_class,
                    int(mvt[j - 1]),
                )
                + np.random.normal(0, noise, int(mvt[j - 1]))
            )

            dx = r * np.cos(t)
            dy = r * np.sin(t)

            dd = np.vstack([dx, dy]).T
            X.append(dd)
            y += [j - 1] * int(mvt[j - 1])

    return np.vstack(X), np.array(y).astype(int)


def load_mnist(
    n_samples=60000,
    reshape=True,
    save_path='./torchvision_datasets',
    train=True,
    random_state=None,
    onehot=False,
):
    """
    By default, 60000 training samples and 10000 test samples
    """
    from torchvision import datasets

    if train and n_samples > 60000:
        from warnings import warn
        warn("MNIST training set has max 60000 samples. Providing 60000 samples.")
    elif not train and n_samples > 10000:
        from warnings import warn
        warn("MNIST test set has max 10000 samples. Providing 10000 samples.")

    dataset = datasets.MNIST(save_path, train=train, download=True)
    X = dataset.data.numpy()[:n_samples]
    if reshape:
        X = X.reshape((X.shape[0], -1))
        

    if onehot:
        y = nn.functional.one_hot(dataset.targets).numpy()[:n_samples]
    else:
        y = dataset.targets.numpy()[:n_samples]

    np.random.seed(random_state)
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def samples_trunk(n_samples, dims=10, random_state=None):
    n = n_samples
    d = dims
    np.random.seed(random_state)
    n1 = np.random.binomial(n, 0.5)
    n2 = n - n1
    mu = 1 / np.sqrt(np.arange(1, d+1))

    X = np.vstack((
        np.random.normal(mu, 1, (n1, d)),
        np.random.normal(-mu, 1,  (n2, d))
    ))
    
    return X, np.asarray([1]*n1 + [0]*n2)
