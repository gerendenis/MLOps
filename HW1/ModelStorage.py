from BaseModel import BaseModel
import pandas as pd
import pickle
import os


class ModelStorage(object):
    """
    Class for operations with models

    save_path : str, Required, path to save models

    """

    def __init__(self, save_path):
        self.current_models = {}

        if not os.path.exists(save_path):
            raise ValueError('Invalid save_path for models.')

        self.save_path = save_path

    def set_model(self, model_id, model_type=None, model_comment=None, new_model=True, rewrite_model=False):
        """
        Generates model object

        Parameters:
        ----------
        model_id : int, Required, id of the model
        new_model : bool, Optional, fit new model or load old model
        model_type : str, Optional, type of the model ('Regression' | 'BinaryClassification').
                        Required if new_model = True
        model_comment : str, Optional, comment
        rewrite_model : bool, Optional, rewrite old model with model_id if exists

        """

        model_path = f'{self.save_path}/model_{model_id}'

        if new_model:

            if not rewrite_model and os.path.exists(f'{self.save_path}/model_{model_id}'):
                raise ValueError(f'Model with id {model_id} exists. To rewrite set "rewrite_model=True"')

            if not model_type:
                raise ValueError(f'Provide model_type for new_model')

            model = BaseModel(model_type)
            self.current_models[model_id] = {'model': model,
                                             'model_type': model_type,
                                             'model_comment': model_comment}

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        if not new_model:
            if not os.path.exists(model_path):
                raise ValueError(f'Model with id {model_id} doesnt exist.')

            with open(model_path, 'rb') as f:

                model = pickle.load(f)
                model_type = model.model_type
                model_comment = model.model_comment

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
            raise ValueError(f'Model with id {model_id} not in loaded models.')
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

        model_path = f'{self.save_path}/model_{model_id}'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def predict_model(self, model_id, data=None):
        """
        Get model predict

        Parameters:
        ----------
        model_id : int, Required, id of the model
        data : np.array, data to predict (if None predicts on train data)
        """

        if model_id not in self.current_models:
            raise ValueError(f'Model with id {model_id} not in loaded models.')
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

        model_path = f'{self.save_path}/model_{model_id}'

        if not os.path.exists(model_path):
            raise ValueError(f'Model with id {model_id} not in models.')

        if model_id in self.current_models:
            self.current_models.pop(model_id)

        os.remove(model_path)

    def get_current_models(self, loaded=True):
        """
        Returns all saved models from save_path

        loaded: bool, Optional, show all models or only loaded
        """

        if loaded:
            models_info = {}

            for model_id in self.current_models:

                model_type = self.current_models[model_id]['model_type']
                is_fitted = self.current_models[model_id]['model'].is_fitted
                comment = self.current_models[model_id]['model_comment']

                if not comment:
                    comment = '-'

                models_info[model_id] = f'Model_type: {model_type}, is_fitted : {is_fitted}, comment : {comment}'

            return models_info

        else:

            all_models = [f for f in os.listdir(self.save_path) if
                          os.path.isfile(os.path.join(self.save_path, f))]
            models_info = {}

            for file in all_models:
                model_path = f'{self.save_path}/{file}'
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                    model_type = model.model_type
                    is_fitted = model.is_fitted
                    comment = model.model_comment

                    if not comment:
                        comment = '-'
                    
                    model_id = int(str(file)[6:])
                    models_info[model_id] = f'Model_type: {model_type}, is_fitted : {is_fitted}, comment : {comment}'

            return models_info
