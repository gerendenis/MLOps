import flask
from flask import jsonify
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from ModelStorage import ModelStorage
import json

app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='MlFlow KEKW',
          description='Тык-пык учим модельки',
          )
all_models = ModelStorage()

set_params = api.model('Set', {
    'model_id': fields.Integer(required=True,
                               description='Model id',
                               example=1),
    'model_type': fields.String(required=True,
                                description='Model type (Regression | BinaryClassification)',
                                example='Regression'),
    'model_comment': fields.String(required=False,
                                   description='Model type (Regression | BinaryClassification)',
                                   example='Comment')})

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

delete_params = api.model('Delete', {
    'model_id': fields.Integer(description='Model id', example=1),
})

ns = api.namespace('Models', description='Тут должно что-то происходить')


@app.errorhandler(Exception)
def handle_error(error):
    message = [str(x) for x in error.args]
    response = {
        'error': {
            'type': error.__class__.__name__,
            'message': message
        }
    }
    return jsonify(response), 500


@ns.route('/models_list')
class ModelsList(Resource):
    def get(self):
        """List all current models"""
        return jsonify(all_models.get_current_models())


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
    app.run(debug=True)
