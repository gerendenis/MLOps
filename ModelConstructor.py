from BaseModel import BaseModel


class ModelConstractor(object):
    """
    Class for operations with models

    """
    def __init__(self):
        self.current_models = {}

    def set_model(self, model_name, model_type, features=None, target=None):
        """
        Generates model object

        Parameters:
        ----------
        model_name : str, name of the model
        model_type : str, type of the model ('Regression' | 'BinaryClassification')
        features : np.array(), data for model (if None generates automatically)
        target : np.array(), target vector for model (if None generates automatically)

        """

        model = BaseModel(model_name, model_type, features, target)
        self.current_models[model_name] = model

    def fit_model(self, model_name, model_params=None, refit=True):
        """
        Fits chosen model

        Parameters:
        ----------
        model_name : str, name of the model
        model_params : dict, input params
        refit : bool, refit existing model
        """

        if model_name not in self.current_models:
            raise ValueError(f'Model {model_name} not in models.')
        else:
            model = self.current_models[model_name]

        if model.is_fitted and not refit:
            raise ValueError(f'Model {model_name} is already fitted and refit param is False')
        else:
            model.fit(model_params=model_params)
            self.current_models[model_name] = model

    def predict_model(self, model_name, data=None):
        """
        Get model predict

        Parameters:
        ----------
        model_name : str, name of the model
        data : np.array, data to predict (if None predicts on train data)
        """

        if model_name not in self.current_models:
            raise ValueError(f'Model {model_name} not in models.')
        else:
            model = self.current_models[model_name]

        predict = model.predict(data)

        return predict

    def delete_model(self, model_name):
        """
        Delete chosen model

        Parameters:
        ----------
        model_name : str, name of the model
        """

        if model_name not in self.current_models:
            raise ValueError(f'Model {model_name} not in models.')
        else:
            current_models = self.current_models
            current_models.pop(model_name)

            self.current_models = current_models

    def get_current_models(self):
        """
        Returns current models dict

        """

        models_info = {}

        for model_name in self.current_models:
            model_type = self.current_models[model_name].model_type
            is_fitted = self.current_models[model_name].is_fitted
            models_info[model_name] = f'Model_type: {model_type}, is_fitted : {is_fitted}'

        return models_info
