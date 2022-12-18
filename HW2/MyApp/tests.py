from BaseModel import BaseModel
from ModelStorage import ModelStorage
from sqlalchemy import create_engine
import pandas as pd


# model test
def test_base_model(model_type='Regression'):
    model = BaseModel(model_type)
    assert model.fit(fit_example=True) is None
    assert model.predict(predict_example=True) is None


test_base_model()


# storage test
def test_storage(path='./models'):
    all_models = ModelStorage(path)

    assert all_models.set_model(model_id=-1,
                                model_type='Regression',
                                model_comment='Test model',
                                new_model=True,
                                rewrite_model=True) is None

    assert all_models.fit_model(model_id=-1,
                                model_params=None,
                                refit=True) is None

    assert all_models.delete_model(model_id=-1) is None


test_storage()


# db test
def create_db():
    engine = create_engine('postgresql://postgres:postgres@db:5432/postgres')
    df = pd.DataFrame([[1, 10, 1],
                       [2, 20, 1],
                       [3, 30, 1]], columns=['target', 'feature', 'const'])
    df.to_sql('df_test', con=engine, if_exists='replace')


create_db()
