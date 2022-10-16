import flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from ModelConstructor import ModelConstractor
import json

app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='MlFlow KEKW',
          description='Тык-пык учим модельки')

all_models = ModelConstractor()

set_params = api.model('Set', {
    'model_name': fields.String(description='Model name',
                                example='Model 1'),
    'model_type': fields.String(description='Model type (Regression | BinaryClassification)',
                                example='Regression')})

fit_params = api.model('Fit', {
    'model_name': fields.String(description='Model name', example='Model 1'),
    'model_params': fields.Arbitrary(description='Model params', example=json.dumps({'max_depth': 3})),
    'refit': fields.Boolean(description='Refit if exists', example=bool(True))})

predict_params = api.model('Predict', {
    'model_name': fields.String(description='Model name')})

delete_params = api.model('Delete', {
    'model_name': fields.String(description='Model name')})

ns = api.namespace('Models', description='Тут должно что-то происходить')
@ns.route('/models_list')
class ModelsList(Resource):
    def get(self):
        """List all current models"""
        return all_models.get_current_models()


@ns.route('/model/set')
class SetModel(Resource):
    @ns.expect(set_params)
    def post(self):
        """Add new model"""
        all_models.set_model(**api.payload)


@ns.route('/model/fit')
class FitModel(Resource):
    @ns.expect(fit_params)
    def post(self):
        """Fit model"""
        all_models.fit_model(**api.payload)


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
        return all_models.delete_model(**api.payload)


if __name__ == '__main__':
    app.run(debug=True)
