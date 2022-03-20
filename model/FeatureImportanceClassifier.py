import numpy as np
import copy as cp
import time

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.feature_selection import mutual_info_classif


class FeatureImportanceClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Feature Importance ensemble classifier.

    Parameters
    ----------
    base_estimators: List of StreamModel or sklearn.BaseEstimator (default=[NaiveBayes(), HoeffdingTreeClassifier())
        Each member of the ensemble is an instance of the base estimator.
    period: int (default=100)
        Period between expert removal, creation, and feature importance check.
    threshold: float (default=0.02)
        Threshold that determines the new training of models
    Notes
    -----
    The feature importance classifier (FIC), uses five mechanisms to
    cope with concept drift: It trains a pool of experts based on defined base estimators,
    it updates the pool of experts when any of the important feature of the variables exceed the threshold
    """
    class ExpertWrapper:
        """
        Wrapper that includes an estimator

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        num_classes: int
            The number of actual target classes
        """
        def __init__(self, estimator, num_classes):
            self.estimator = estimator
            self.num_classes = num_classes

    def __init__(self, base_estimators=[NaiveBayes(), HoeffdingTreeClassifier()],
                 period=1000, threshold=0.2):
        """
        Creates a new instance of FeatureImportanceClassifier.
        """
        super().__init__()
        self.base_estimators = base_estimators
        self.period = period
        self.threshold = threshold

        self.p = -1  # chunk pointer
        self.window_size = None  # chunk size
        self.epochs = None
        self.num_classes = None
        self.experts = None
        self.X_batch = None
        self.y_batch = None

        self.custom_measurements = []

        self.reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Returns
        -------
        FeatureImportanceClassifierClassifier
            self
        """
        for i in range(len(X)):
            self.fit_single_sample(X[i:i + 1, :], y[i:i + 1], classes, sample_weight)
        return self

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Fits a single sample of shape `X.shape=(1, n_attributes)` and `y.shape=(1)`

        Retrains each expert on the provided batch of data if any of
        the important feature of the variables exceed the threshold

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: list
            List of all existing classes. This is an optional parameter.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Applicability
            depends on the base estimator.

        """
        start_time = time.time()

        if self.p <= 0:
            for exp in self.experts:
                exp.estimator = self.train_model(cp.copy(exp.estimator), X, y, classes, sample_weight)

        N, D = X.shape
        self.window_size = self.period

        if self.p <= 0:
            self.X_batch = np.zeros((self.window_size, D), dtype=int)
            self.y_batch = np.zeros(self.window_size, dtype=int)
            self.p = 0

        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)

        self.X_batch[self.p] = X
        self.y_batch[self.p] = y
        self.p = self.p + 1
        self.epochs += 1

        if self.p >= self.window_size:
            self.p = 0
            feat_importance = self.calculate_feature_importance(self.X_batch, self.y_batch)
            if any([fi >= self.threshold for fi in feat_importance]):
                # retrain models
                for exp in self.experts:
                    exp.estimator = self.train_model(exp.estimator, self.X_batch, self.y_batch, classes, sample_weight)
            data = {'id_period': self.epochs / self.period, 'n_experts': len(self.experts),
                    'feat_importance': feat_importance, 'train_time': (time.time() - start_time)}
            self.custom_measurements.append(data)

    @staticmethod
    def train_model(model, X, y, classes=None, sample_weight=None):
        """ Trains a model, taking care of the fact that either fit or partial_fit is implemented
        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The model to train
        X: numpy.ndarray of shape (n_samples, n_features)
            The data chunk
        y: numpy.array of shape (n_samples)
            The labels in the chunk
        classes: list or numpy.array
            The unique classes in the data chunk
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        Returns
        -------
        StreamModel or sklearn.BaseEstimator
            The trained model
        """
        try:
            model.partial_fit(X, y, classes, sample_weight)
        except NotImplementedError:
            model.fit(X, y)
        return model

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts, n_samples)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def predict(self, X):
        """ predict

        The predict function will take an average of the predictions of its
        learners, weighted by equally weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        predictions_class = np.zeros((len(X), self.num_classes))
        for exp in self.experts:
            Y_hat = exp.estimator.predict(X)
            for i, y_hat in enumerate(Y_hat):
                predictions_class[i][y_hat] += 1/len(self.experts)
        y_hat_final = np.argmax(predictions_class, axis=1)
        return y_hat_final

    def predict_proba(self, X):
        raise NotImplementedError

    @staticmethod
    def calculate_feature_importance(X, y):
        """
        Estimate mutual information for a discrete target variable.
        """
        feat_importance = mutual_info_classif(X, np.ravel(y), n_neighbors=3)
        return feat_importance

    def _construct_new_expert(self, estimator):
        """
        Constructs a new WeightedExpert randomly from list of provided base_estimators.
        """
        return self.ExpertWrapper(cp.deepcopy(estimator), self.num_classes)

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2    # Minimum of 2 classes
        self.experts = []
        for estimator in self.base_estimators:
            self.experts.append(self._construct_new_expert(estimator))
