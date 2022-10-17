from BaseModel import BaseModel
import pandas as pd


class ModelStorage(object):
    """
    Class for operations with models

    """

    def __init__(self):
        self.current_models = {}

    def set_model(self, model_id, model_type, model_comment=None):
        """
        Generates model object

        Parameters:
        ----------
        model_id : int, Required, id of the model
        model_type : str, Required, type of the model ('Regression' | 'BinaryClassification')
        model_comment : str, Optional, comment
        """

        model = BaseModel(model_type)
        self.current_models[model_id] = {'model': model,
                                         'model_type': model_type,
                                         'model_comment': model_comment}

    def fit_model(self, model_id,
                  model_params=None,
                  refit=True,
                  data_path=None,
                  features_list=None,
                  target_variable=None):
        """
        Fits chosen model

        Parameters:
        ----------
        model_id : int, Required, id of the model
        model_params : dict, input params
        refit : bool, refit existing model
        data_path : str, Path to data file
        features_list : list, features for fit method
        target_variable : str, target variable label
        """

        if model_id not in self.current_models:
            raise ValueError(f'Model with id {model_id} not in models.')
        else:
            model = self.current_models[model_id]['model']

        if model.is_fitted and not refit:
            raise ValueError(f'Model {model_id} is already fitted and refit param is False')

        # write constraints
        if data_path:
            all_data = pd.read_csv(data_path)
            data = all_data[features_list]
            target = all_data[target_variable]
            model.fit(data=data, target=target, model_params=model_params, fit_example=False)
        else:
            model.fit(model_params=model_params, fit_example=True)

        self.current_models[model_id]['model'] = model

    def predict_model(self, model_id, data=None):
        """
        Get model predict

        Parameters:
        ----------
        model_id : int, Required, id of the model
        data : np.array, data to predict (if None predicts on train data)
        """

        if model_id not in self.current_models:
            raise ValueError(f'Model with id {model_id} not in models.')
        else:
            model = self.current_models[model_id]['model']

        predict = model.predict(data)

        return predict

    def delete_model(self, model_id):
        """
        Delete chosen model

        Parameters:
        ----------
        model_id : int, Required, id of the model
        """

        if model_id not in self.current_models:
            raise ValueError(f'Model with id {model_id} not in models.')
        else:
            current_models = self.current_models
            current_models.pop(model_id)

            self.current_models = current_models

    def get_current_models(self):
        """
        Returns current models dict

        """

        models_info = {}

        for model_id in self.current_models:

            model_type = self.current_models[model_id]['model_type']
            is_fitted = self.current_models[model_id]['model'].is_fitted
            comment = self.current_models[model_id]['model_comment']

            if not comment:
                comment = '-'

            models_info[model_id] = f'Model_type: {model_type}, is_fitted : {is_fitted}, comment : {comment}'

        return models_info

# features : np.array(), data for model (if None generates automatically)
# target : np.array(), target vector for model (if None generates automatically)
