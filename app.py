from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():

    return render_template('form.html')

@app.route('/status',methods=['POST'])
def status():
    name = request.form.get('name')
    dependents = request.form.get('dependents')
    education = request.form.get('education')
    employed = request.form.get('self')
    income = request.form.get('income')
    loan_amount = request.form.get('loanamount')
    loan_term = request.form.get('loanterm')
    cibil_score = request.form.get('cibil')
    resedential_assets = request.form.get('residential')
    commercial_assets = request.form.get('commercial')
    bank_assets = request.form.get('bank')

    data = [dependents, income, loan_amount, loan_term, cibil_score, resedential_assets, commercial_assets, bank_assets]
    
    data_array = np.array(data).reshape(-1,1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)

    if employed == 'yes':
        employed = 1
    else:
        employed = 0

    if education == 'Graduate':
        education = 0
    else:
        education = 1

    final_data = np.append(scaled_data, np.array([education, employed]))

    with open('lr_model.pkl','rb') as file:
        loaded_model = pickle.load(file)

    #model = pickle.load(open('lr_model.pkl', 'rb'))

    result = loaded_model.predict([final_data])

    print(result)
    return render_template('status.html')

if __name__ == '__main__':
    app.run(debug=True)
