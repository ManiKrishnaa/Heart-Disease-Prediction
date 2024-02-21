from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        p_age = float(request.form['age'])
        p_sex = 1 if request.form['sex'] == 'male' else 0
        p_cp = request.form['cp']
        if p_cp == 'typical_angina':
            p_cp = 0
        elif p_cp == 'atypical_angina':
            p_cp = 1
        elif p_cp == 'non_anginal_pain':
            p_cp = 2
        elif p_cp == 'asymptomatic':
            p_cp = 3
        p_rbp = float(request.form['restBloodPressure'])
        p_chl = float(request.form['cholesterol'])
        p_fbs = float(request.form['fastingBloodSugar'])
        if p_fbs > 120:
            p_fbs = 1
        else:
            p_fbs = 0
        p_recg = {'normal': 0, 'st_t_abnormality': 1, 'lvh': 2}[request.form['ecgResults']]
        p_maxhr = float(request.form['maxHeartRate'])
        p_eia = 1 if request.form['exerciseAngina'] == 'present' else 0
        p_oldpeak = float(request.form['oldPeak'])
        p_slope = {'unsloping': 1, 'flat': 2, 'downsloping': 3}[request.form['slope']]
        p_ca = float(request.form['ca'])
        p_thal = {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2}[request.form['thalassemia']]

        print(p_age,p_sex,p_cp,p_rbp,p_chl,p_fbs,p_recg,p_maxhr,p_eia,p_oldpeak,p_slope,p_ca,p_thal)
        with open('heart_disease_trained.pkl', 'rb') as file:
            model = pickle.load(file)

        x = np.array([[p_age, p_sex, p_cp, p_rbp, p_chl, p_fbs, p_recg, p_maxhr, p_eia, p_oldpeak, p_slope, p_ca, p_thal]])
        X_df = pd.DataFrame(x, columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

        ans = model.predict(X_df)

        prediction = "Positive" if ans[0] == 1 else "Negative"
        app.logger.info(f"Sending response: {prediction}")
            
        return jsonify(prediction)
    except Exception as e:
        app.logger.error(f"Error processing data: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the data.'})

if __name__ == '__main__':
    app.run(debug=True)
