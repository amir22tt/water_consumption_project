# Flask web app: Water Consumption Predictor (app.py)
# This file expects templates/index.html and templates/about.html to exist in templates/.

from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret"

MODEL_PATH = model_path = "model.pkl"
  # updated model filename
model = None

# Load unique values from dataset
try:
    df_data = pd.read_csv('processed_data.csv')
    PERSON_IDS = sorted(df_data['Person_ID'].unique().tolist())
    GENDERS = sorted(df_data['Gender'].unique().tolist())
    CITIES = sorted(df_data['City'].unique().tolist())
    ACTIVITY_LEVELS = sorted(df_data['Activity_Level'].unique().tolist())
except:
    PERSON_IDS = ['P0001']
    GENDERS = ['Female', 'Male']
    CITIES = ['Karachi']
    ACTIVITY_LEVELS = ['Low', 'Medium', 'High']


def try_load_model(path=MODEL_PATH):
    global model
    if model is not None:
        return model
    try:
        if os.path.exists(path):
            model_obj = joblib.load(path)
            print(f"Loaded model from {path}")
            model = model_obj
            return model
        else:
            print(f"Model not found at {path}")
            return None
    except Exception as e:
        print('Error loading model:', e)
        return None


def fallback_predict(row):
    temp = float(row.get('Temperature_C', 25))
    activity = row.get('Activity_Level', 'Medium')
    act_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
    base = 1.6
    temp_effect = max(0, (temp - 20) * 0.05)
    activity_effect = act_map.get(activity, 0.5) * 1.0
    predicted = round(base + temp_effect + activity_effect, 2)
    return predicted


@app.route('/', methods=['GET'])
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    try_load_model()
    return render_template('index.html',
                           person_ids=PERSON_IDS,
                           genders=GENDERS,
                           cities=CITIES,
                           activity_levels=ACTIVITY_LEVELS,
                           today=today,
                           model_path=MODEL_PATH,
                           last_prediction=None,
                           model_used=("Loaded" if model is not None else "Fallback"),
                           form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Date': request.form.get('date'),
        'Person_ID': request.form.get('person_id'),
        'Age': request.form.get('age'),
        'Gender': request.form.get('gender'),
        'City': request.form.get('city'),
        'Temperature_C': request.form.get('temperature'),
        'Activity_Level': request.form.get('activity_level')
    }
    df = pd.DataFrame([data])

    mdl = try_load_model()
    predicted_value = None
    model_used = 'Fallback'
    if mdl is not None:
        try:
            preds = mdl.predict(df)
            if isinstance(preds, (list, tuple, np.ndarray)):
                predicted_value = float(preds[0])
            else:
                predicted_value = float(preds)
            model_used = type(mdl).__name__
        except Exception as e:
            print('Model predict failed:', e)
            predicted_value = fallback_predict(data)
            model_used = 'Fallback'
    else:
        predicted_value = fallback_predict(data)

    predicted_value = round(float(predicted_value), 2)
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html',
                           person_ids=PERSON_IDS,
                           genders=GENDERS,
                           cities=CITIES,
                           activity_levels=ACTIVITY_LEVELS,
                           today=today,
                           model_path=MODEL_PATH,
                           last_prediction=predicted_value,
                           model_used=model_used,
                           form_data=data)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash('Failed to read CSV: ' + str(e))
        return redirect(url_for('index'))

    required_cols = ['Date','Person_ID','Age','Gender','City','Temperature_C','Activity_Level']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        flash('CSV missing columns: ' + ','.join(missing))
        return redirect(url_for('index'))

    mdl = try_load_model()
    if mdl is not None:
        try:
            preds = mdl.predict(df)
            df['Predicted_Water_Liters'] = preds
        except Exception as e:
            print('Model batch predict failed:', e)
            df['Predicted_Water_Liters'] = df.apply(lambda r: fallback_predict(r.to_dict()), axis=1)
    else:
        df['Predicted_Water_Liters'] = df.apply(lambda r: fallback_predict(r.to_dict()), axis=1)

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return send_file(io.BytesIO(out.getvalue().encode('utf-8')),
                     mimetype='text/csv',
                     attachment_filename='predictions.csv',
                     as_attachment=True)


@app.route('/about')
def about():
    return render_template('about.html', model_path=MODEL_PATH)


if __name__ == '__main__':
    try_load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
