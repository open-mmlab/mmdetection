"""
This script serves for testing on processing the request and returning the prediction in string.
"""
from flask import Flask, request, jsonify
from app.config import cfg  # since app is the base package
from app.mmdet_utils import get_prediction  # since app is the base package
import pathlib2

current_dir = pathlib2.Path.cwd()
project_dir = current_dir.parent
ds_path = project_dir / 'test_imgs'
print(f"the path to the folder holding the testing imgs: {ds_path}")

app = Flask(__name__)

def allowed_file(filename):
    # e.g., xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in cfg.ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        # not a very good error handling
        try:
            prediction = get_prediction(cfg.CONFIG,
                                        cfg.CHECKPOINT,
                                        (ds_path / '7ae19de7bc2a.png').as_posix())
            # TODO: now only test the flask function, so the result is only temporary.
            resp_data = {'image_id': prediction.loc[0, 'id'], 'prediction': prediction.loc[0, 'predicted']}
            return jsonify(resp_data)
        except:
            return jsonify({'error': 'error during prediction'})

if __name__ == '__main__':
    app.run(debug=True)