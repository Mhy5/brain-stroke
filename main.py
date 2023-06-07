import tensorflow as tf;
import numpy as np;
from PIL import Image
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib


from flask import request
from flask import Flask


app = Flask(__name__)

image_model = tf.keras.models.load_model('stroke-model.h5')
text_model = joblib.load("randomForest")
scaler = joblib.load("randomForestScaler")


@app.route('/stroke-image', methods=['POST'])
def image_predict():
    if request.method == 'POST':
        f = request.files['image']
        img = Image.open(f.stream)
        new_img = img.convert("RGB").resize((224,224))
        # convert to array
        img_array = np.asarray(new_img)
        tensor = tf.convert_to_tensor(img_array)
        tensor = tf.expand_dims(tensor, axis=0)
        results = image_model.predict(tensor)[0].tolist()
        return {
            "normal": results[0],
            "stroke":results[1]
        }
    return {
        "error":"Unvalid Request"
    }

@app.route('/stroke-text', methods=['POST'])
def text_predict():
    if request.method == 'POST':
        entered_data = 	[[
        request.form.get("gender", type=float),
        request.form.get("age",type=float),
        request.form.get("hypertension",type=float),
        request.form.get("heart_disease",type=float),
        request.form.get("ever_married",type=float),
        request.form.get("work_type",type=float),
        request.form.get("Residence_type",type=float),
        request.form.get("avg_glucose_level",type=float),
        request.form.get("bmi",type=float),
        request.form.get("smoking_status",type=float)]]
        results = text_model.predict(scaler.transform(entered_data))
        return {
            "stroke": True if results[0] else False
        }
