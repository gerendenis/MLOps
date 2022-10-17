from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def make_example_data(model_type, n_samples=100, n_features=5):
    """
    Generates data for chosen model

    Parameters:
    ----------
    model_type : str, type of the model ('Regression' | 'BinaryClassification')
    n_samples : int, The number of samples.
    n_features : int, The number of features.

    """
    data, target = None, None

    if model_type == 'Regression':
        data, target = make_regression(n_samples=n_samples,
                                       n_features=n_features)

    if model_type == 'BinaryClassification':
        data, target = make_classification(n_samples=n_samples,
                                           n_features=n_features,
                                           n_classes=2)

    return data, target


class BaseModel:
    """
    Base class for models

    Parameters:
    ----------
    model_type : str, Required, type of the model ('Regression' | 'BinaryClassification')
    model_comment : str, Optional, comment

    """

    def __init__(self, model_type):

        self.model_type = model_type

        self.model = None
        self.fitted_model = None

        self.base_models = {
            'Regression': RandomForestRegressor,
            'BinaryClassification': RandomForestClassifier
        }

        if model_type not in self.base_models:
            raise ValueError(f'Unrecognized model type. Available types: {list(self.base_models.keys())}')

        self.model_params = None
        self.is_fitted = False

    @staticmethod
    def check_params(model, model_params, model_type):
        """
        Checks params in input and model

        Parameters:
        ----------
        model : chosen model
        model_params : dict, input params
        model_type : str, type of the model ('Regression' | 'BinaryClassification')

        """
        all_params = model.get_params().keys()
        bad_params = []

        for param in model_params:
            if param not in all_params:
                bad_params.append(param)
        if len(bad_params) != 0:
            raise ValueError(f'Unrecognized params for model {model_type}: {bad_params}')

    def fit(self, data=None, target=None, model_params=None, fit_example=True):
        """
        Fits model

        Parameters:
        ----------
        data : vector, data to fit on
        target : vector, target variable
        model_params : dict, input params
        fit_example : bool, if True fit on generated data
        """

        if ((not data or not target) and not fit_example) or ((data and target) and fit_example):
            raise ValueError(f'Wrong fit specification. Provide data and target or set '
                             f'fit_example=True')

        if fit_example:
            data, target = make_example_data(self.model_type)
        self.model = self.base_models[self.model_type]()

        if model_params is None:
            model_params = {}

        if len(model_params) > 0:
            self.model_params = model_params
            self.check_params(self.model, self.model_params, self.model_type)
            self.model = self.model.set_params(**self.model_params)

        self.fitted_model = self.model.fit(data, target)
        self.is_fitted = True

    def predict(self, data=None, predict_example=True):
        """
        Get model predict

        Parameters:
        ----------
        data : np.array, data to predict (if None predicts on self.features)
        predict_example : bool, if True predict on generated data

        """
        if not self.is_fitted:
            raise ValueError('Model is not fitted. Try .fit() method :)')

        if data and predict_example:
            raise ValueError(f'Wrong predict specification. Provide data or set'
                             f'predict_example=True')

        if predict_example:
            data, _ = make_example_data(self.model_type)

        predict = self.fitted_model.predict(data)

        return predict
