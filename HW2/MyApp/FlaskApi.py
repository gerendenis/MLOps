import flask
from flask import jsonify
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from ModelStorage import ModelStorage
import json
from sqlalchemy import create_engine
import pandas as pd

app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='MlFlow KEKW',
          description='Тык-пык учим модельки',
          )

# проверяем, что есть коннект к бд
def create_db():
    engine = create_engine('postgresql://postgres:postgres@db:5432/postgres')
    df = pd.DataFrame([[1, 10, 1],
                   [2, 20, 1],
                   [3, 30, 1]], columns=['target', 'feature', 'const'])
    df.to_sql('df_test', con=engine, if_exists='replace')

create_db()


set_params = api.model('Set', {
    'model_id': fields.Integer(required=True,
                               description='Model id',
                               example=1),
    'model_type': fields.String(required=True,
                                description='Model type (Regression | BinaryClassification)',
                                example='Regression'),
    'model_comment': fields.String(required=False,
                                   description='Model type (Regression | BinaryClassification)',
                                   example='Comment'),
    'new_model': fields.Boolean(required=False, description='new_model', example=bool(True)),
    'rewrite_model': fields.Boolean(required=False, description='rewrite if exists', example=bool(False))})


fit_params = api.model('Fit', {
    'model_id': fields.Integer(required=True, description='Model id', example=1),
    'model_params': fields.Arbitrary(required=False, description='Model params', example=json.dumps({'max_depth': 3})),
    'refit': fields.Boolean(required=False, description='Refit if exists', example=bool(True)),

    'data_path': fields.String(required=False, description='Path to data file'),
    'features_list': fields.String(required=False, description='Features for fit method'),
    'target_variable': fields.String(required=False, description='Target variable label'),
})

predict_params = api.model('Predict', {
    'model_id': fields.Integer(required=True, description='Model id', example=1),
    'save_path': fields.String(required=False, description='Path to save predictions.'),

})

models_list_params = api.model('ModelsList', {
    'loaded': fields.Boolean(required=False)})

delete_params = api.model('Delete', {
    'model_id': fields.Integer(description='Model id', example=1),
})

ns = api.namespace('Models', description='Тут должно что-то происходить')


@app.errorhandler(Exception)
def handle_error(error):
    status_code = 500

    if status_code in dir(error):
        status_code = error.status_code

    message = [str(x) for x in error.args]
    response = {
        'error': {
            'type': error.__class__.__name__,
            'message': message
        }
    }
    return jsonify(response), status_code


all_models = ModelStorage('./models')


@ns.route('/models_list/get')
class ModelsList(Resource):
    @ns.expect(models_list_params)
    def get(self):
        """List all current models"""
        return jsonify(all_models.get_current_models(**api.payload))


@ns.route('/model/set')
class SetModel(Resource):
    @ns.expect(set_params)
    def post(self):
        """Add new model"""
        all_models.set_model(**api.payload)
        return jsonify({'message': f"Model {api.payload['model_id']} successfully set."})


@ns.route('/model/fit')
class FitModel(Resource):
    @ns.expect(fit_params)
    def post(self):
        """Fit model"""
        all_models.fit_model(**api.payload)
        return jsonify({'message': f"Model {api.payload['model_id']} successfully fitted."})


@ns.route('/model/predict')
class PredictModel(Resource):
    @ns.expect(predict_params)
    def get(self):
        """Get model predict"""
        return json.dumps({"predict": all_models.predict_model(**api.payload)
                          .astype(float)
                          .tolist()})


@ns.route('/model/delete')
class DeleteModel(Resource):
    @ns.expect(delete_params)
    def delete(self):
        """Delete model"""
        all_models.delete_model(**api.payload)
        return jsonify({'message': f"Model {api.payload['model_id']} successfully deleted."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    