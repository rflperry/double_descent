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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import zero_one_loss


class ReluNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_dims=[100, 100],
        n_epochs=100,
        learning_rate=0.01,
        batch_size=128,
        shuffle=True,
        callbacks=[],
        use_gpu=False,
        verbose=0,
        early_stop_thresh=0,
    ):

        self.history_ = None
        self.model_ = None
        self.gpu_ = use_gpu and torch.cuda.is_available()

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        # Sets all attributes from the initialization
        for arg, val in values.items():
            setattr(self, arg, val)
        self.hidden_layer_dims = list(self.hidden_layer_dims)

    def _build_model(self):
        self._layer_dims = [self.input_dim_] + self.hidden_layer_dims + [
            self.output_dim_]

        self.model_ = torch.nn.Sequential()

        # Loop through the layer dimensions and create an input layer, then
        # create each hidden layer with relu activation.
        for idx, dim in enumerate(self._layer_dims):
            if idx < len(self._layer_dims) - 1:
                module = torch.nn.Linear(dim, self._layer_dims[idx + 1])
                init.xavier_uniform_(module.weight)
                self.model_.add_module("linear" + str(idx), module)

            if idx < len(self._layer_dims) - 2:
                self.model_.add_module("relu" + str(idx), torch.nn.ReLU())

        if self.gpu_:
            self.model_ = self.model_.cuda()

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

        loss_fn = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self.history_ = {"bce_loss": [], "01_error": []}

        finish = False
        for epoch in range(self.n_epochs):
            if finish:
                break

            loss = None
            idx = 0
            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self.model_(Variable(minibatch))

                loss = loss_fn(
                    y_pred,
                    Variable(target.cuda().float() if self.gpu_ else target.float()),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_labels = target.cpu().numpy() if self.gpu_ else target.numpy()
            y_pred_results = torch.nn.functional.softmax(
                y_pred.cpu().detach() if self.gpu_ else y_pred.detach(), dim=1
            ).numpy()

            error = zero_one_loss(y_pred_results.argmax(1), y_labels.argmax(1))
            if error <= self.early_stop_thresh:
                finish = True

            self.history_["01_error"].append(error)
            self.history_["bce_loss"].append(loss.detach().item())

            if self.verbose > 0 and epoch % 5 == 0:
                print(
                    "Results for epoch {}, bce_loss={:.2f}, 01_error={:.2f}".format(
                        epoch + 1, loss.detach().item(), error
                    )
                )
            for callback in self.callbacks:
                callback.call(self.model_, self.history_)
                if callback.finish:
                    finish = True
                    break

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

        y = self._reshape_targets(y)
        self.input_dim_ = X.shape[1]
        self.output_dim_ = y.shape[1]
        
        self._build_model()
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
        for batch in np.array_split(X, split_size):
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self.model_(x_pred.cuda() if self.gpu_ else x_pred)
            y_pred_formatted = torch.nn.functional.softmax(
                y_pred.cpu().detach() if self.gpu_ else y_pred.detach(), dim=1
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

    @property
    def n_parameters_(self):
        return sum(p.numel() for p in self.model_.parameters())

    def get_internal_representation(self, X, penultimate=True):
        """
        Returns the internal reprensetation matrix, encoding which samples
        activate which ReLUs. If penultimate=True, only returns the matrix
        for the penultimate layer.
        """
        if self.history_ is None:
            raise RuntimeError("Classifier has not been fit")

        split_size = math.ceil(len(X) / self.batch_size)
        
        irm = []
        for batch in np.array_split(X, split_size):
            batch_irm = []
            x_pred = Variable(torch.from_numpy(batch).float())
            for module in next(self.model_.modules()):
                x_pred = module(x_pred.detach())
                if type(module) == torch.nn.modules.activation.ReLU:
                    batch_irm.append((x_pred.detach().numpy() > 0).astype(int))
            if penultimate:
                irm.append(batch_irm[-1])
            else:
                irm.append(np.hstack(batch_irm))
        
        return np.vstack(irm)
