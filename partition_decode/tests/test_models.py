from partition_decode.models import ReluNetClassifier
import numpy as np


def test_relu_net_runs():
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 4))
    y = np.random.choice(2, (100))
    n_classes = len(np.unique(y))
    net = ReluNetClassifier(num_epochs=1)
    net.fit(X, y)

    y_hat = net.predict(X, y)
    assert y_hat.shape == y.shape, f'{y.shape} != {y_hat.shape}'
    assert len(np.unique(y_hat)) <= n_classes

    y_hat_proba = net.predict_proba(X, y)
    assert y_hat_proba.shape == (len(y), n_classes), f'{y.shape} != {y_hat_proba.shape}'


def test_relu_net_irm():
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 4))
    y = np.random.choice(2, (100))
    hidden_layer_dims = [10, 15]
    net = ReluNetClassifier(num_epochs=1, hidden_layer_dims=hidden_layer_dims)
    net.fit(X, y)

    irm = net.get_internal_representation(X, penultimate=True)
    assert irm.shape == (X.shape[0], hidden_layer_dims[-1])
    irm = net.get_internal_representation(X, penultimate=False)
    assert irm.shape == (X.shape[0], sum(hidden_layer_dims))


test_relu_net_runs()
test_relu_net_irm()
