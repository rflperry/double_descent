"""
Neural network models
"""

import inspect
import math

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import zero_one_loss, mean_squared_error


class ReLuNet(BaseEstimator):
    def __init__(
        self,
        hidden_layer_dims=[100, 100],
        n_epochs=100,
        learning_rate=0.01,
        batch_size=32,
        shuffle=True,
        callbacks=[],
        use_gpu=False,
        verbose=0,
        early_stop_thresh=None,
        bias=True,
    ):

        assert isinstance(hidden_layer_dims, (list, np.ndarray))

        self.history_ = None
        self.model_ = None
        self.gpu_ = use_gpu and torch.cuda.is_available()

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        # Sets all attributes from the initialization
        for arg, val in values.items():
            setattr(self, arg, val)
        self.hidden_layer_dims = list(self.hidden_layer_dims)

    def _build_model(self, input_dim, output_dim):
        self._layer_dims = [input_dim] + self.hidden_layer_dims + [output_dim]

        self.model_ = torch.nn.Sequential()

        # Loop through the layer dimensions and create an input layer, then
        # create each hidden layer with relu activation.
        for idx, dim in enumerate(self._layer_dims):
            if idx < len(self._layer_dims) - 1:
                module = torch.nn.Linear(dim, self._layer_dims[idx + 1], bias=self.bias)
                init.xavier_uniform_(module.weight)
                self.model_.add_module("linear" + str(idx), module)

            if idx < len(self._layer_dims) - 2:
                self.model_.add_module("relu" + str(idx), torch.nn.ReLU())

        if self.gpu_:
            self.model_ = self.model_.cuda()

    @property
    def n_parameters_(self):
        return sum(p.numel() for p in self.model_.parameters())

    def get_internal_representation(self, X, penultimate=False):
        """
        Returns the internal reprensetation matrix, encoding which samples
        activate which ReLUs. If penultimate=True, only returns the matrix
        for the penultimate layer.
        """
        if self.history_ is None:
            raise RuntimeError("Classifier has not been fit")

        split_size = math.ceil(len(X) / self.batch_size)

        irm = []
        with torch.no_grad():
            for batch in np.array_split(X, split_size):
                batch_irm = []
                x_pred = Variable(torch.from_numpy(batch).float())
                for module in next(self.model_.modules()):
                    x_pred = module(x_pred)
                    if type(module) == torch.nn.modules.activation.ReLU:
                        batch_irm.append((x_pred.numpy() > 0).astype(int))
                if penultimate:
                    irm.append(batch_irm[-1])
                else:
                    irm.append(np.hstack(batch_irm))

        return np.vstack(irm)


class ReluNetClassifier(ReLuNet, ClassifierMixin):
    def _train_model(self, X, y):
        torch_x = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()
        if self.gpu_:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = data_utils.TensorDataset(torch_x, torch_y)
        train_loader = data_utils.DataLoader(
            train, batch_size=self.batch_size, shuffle=self.shuffle
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        self.model_.train()
        # optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(
            self.model_.parameters(), lr=self.learning_rate, momentum=0.95
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        self.history_ = {"CrossEntropyLoss": [], "01_error": []}

        finish = False
        for epoch in range(self.n_epochs):
            if finish:
                break

            loss = None

            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self.model_(Variable(minibatch))

                loss = loss_fn(
                    y_pred,
                    Variable(target.cuda().long() if self.gpu_ else target.long()),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            y_labels = (target.cpu() if self.gpu_ else target).numpy()
            y_pred_results = (
                (y_pred.cpu().detach() if self.gpu_ else y_pred.detach())
                .numpy()
                .argmax(1)
            )

            error = zero_one_loss(y_pred_results, y_labels)
            total_error = zero_one_loss(self.predict(X), y)

            if (
                self.early_stop_thresh is not None
                and total_error <= self.early_stop_thresh
            ):
                finish = True

            self.history_["01_error"].append(error)
            self.history_["CrossEntropyLoss"].append(loss.detach().item())

            if self.verbose > 0 and epoch % 5 == 0:
                print(
                    f"Results for epoch {epoch + 1}, CrossEntropyLoss={loss.detach().item():.2f}, 01_error={error:.2f}".format()
                )
            for callback in self.callbacks:
                callback.call(self.model_, self.history_)
                if callback.finish:
                    finish = True
                    break

        self.model_.eval()

    def _reshape_targets(self, y):
        """
        Reshapes the targets to the input size
        """
        classes, inverses = np.unique(y, axis=0, return_inverse=True)
        y_mat = np.zeros((y.shape[0], len(classes)))
        y_mat[np.arange(y_mat.shape[0]), inverses] = 1

        return y_mat

    def fit(self, X, y):
        """
        Trains the pytorch ReLU network classifier
        """
        self._build_model(X.shape[1], len(np.unique(y)))

        # y = self._reshape_targets(y)
        self._train_model(X, y)

        return self

    def predict_proba(self, X, y=None):
        """
        Predicts class probabilities using the trained pytorch model
        """
        if self.history_ is None:
            raise RuntimeError("Classifier has not been fit")

        results = []
        split_size = math.ceil(len(X) / self.batch_size)

        # In case the requested size of prediction is too large for memory (especially gpu)
        # split into batchs, roughly similar to the original training batch size. Not
        # particularly scientific but should always be small enough.
        with torch.no_grad():
            for batch in np.array_split(X, split_size):
                x_pred = Variable(torch.from_numpy(batch).float())
                y_pred = self.model_(x_pred.cuda() if self.gpu_ else x_pred)
                y_pred_formatted = torch.nn.functional.softmax(
                    y_pred.cpu() if self.gpu_ else y_pred, dim=1
                ).numpy()
                results += [y_pred_formatted]

        return np.vstack(results)

    def predict(self, X, y=None):
        """
        Makes a class prediction using the trained model
        """
        if self.history_ is None:
            raise RuntimeError("Classifier has not been fit")

        return self.predict_proba(X, y).argmax(1)

    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current
        implementation returns negative zero_one_loss.
        """
        y_pred = self.predict(X, y)
        return zero_one_loss(y, y_pred)


class ReluNetRegressor(ReLuNet, RegressorMixin):
    def _train_model(self, X, y):
        torch_x = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()
        if self.gpu_:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = data_utils.TensorDataset(torch_x, torch_y)
        train_loader = data_utils.DataLoader(
            train, batch_size=self.batch_size, shuffle=self.shuffle
        )

        loss_fn = torch.nn.MSELoss()
        self.model_.train()
        # optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.SGD(
            self.model_.parameters(), lr=self.learning_rate, momentum=0.95
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.n_epochs // 12, gamma=0.1)

        self.history_ = {"MSELoss": [], "mse": []}

        finish = False
        for epoch in range(self.n_epochs):
            if finish:
                break

            loss = None

            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self.model_(Variable(minibatch))

                loss = loss_fn(
                    y_pred,
                    Variable(target.cuda().float() if self.gpu_ else target.float()),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            y_labels = (target.cpu() if self.gpu_ else target).numpy()
            y_pred_results = (
                y_pred.cpu().detach() if self.gpu_ else y_pred.detach()
            ).numpy()

            error = mean_squared_error(y_pred_results, y_labels)
            total_error = mean_squared_error(self.predict(X), y)

            if (
                self.early_stop_thresh is not None
                and total_error <= self.early_stop_thresh
            ):
                finish = True

            self.history_["mse"].append(error)
            self.history_["MSELoss"].append(loss.detach().item())

            if self.verbose > 0 and epoch % 5 == 0:
                print(
                    f"Results for epoch {epoch + 1}, MSELoss={loss.detach().item():.2f}, mse={error:.2f}".format()
                )
            for callback in self.callbacks:
                callback.call(self.model_, self.history_)
                if callback.finish:
                    finish = True
                    break

        self.model_.eval()

    def fit(self, X, y):
        """
        Trains the pytorch ReLU network classifier
        """
        self._build_model(X.shape[1], y.shape[1])
        self._train_model(X, y)

        return self

    def predict(self, X, y=None):
        """
        Predicts targets using the trained pytorch model
        """
        if self.history_ is None:
            raise RuntimeError("Classifier has not been fit")

        results = []
        split_size = math.ceil(len(X) / self.batch_size)

        # In case the requested size of prediction is too large for memory (especially gpu)
        # split into batchs, roughly similar to the original training batch size. Not
        # particularly scientific but should always be small enough.
        with torch.no_grad():
            for batch in np.array_split(X, split_size):
                x_pred = Variable(torch.from_numpy(batch).float())
                y_pred = (self.model_(x_pred.cuda() if self.gpu_ else x_pred)).numpy()
                results += [y_pred]

        return np.vstack(results)

    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current
        implementation returns mean squared error.
        """
        y_pred = self.predict(X, y)
        return mean_squared_error(y, y_pred)
