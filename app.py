import flask
from flask import render_template
import pickle
import pandas
import numpy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv('data/ebw_data.csv')
df = df.drop_duplicates()
df['U'] = 20
df['Q'] = df.IW * df.U
X = df.drop(['Depth', 'Width'], axis=1).values
scaler = StandardScaler()
scaler.fit(X)

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        _IW = float(flask.request.form['_IW'])
        _IF = float(flask.request.form['_IF'])
        _VW = float(flask.request.form['_VW'])
        _FP = float(flask.request.form['_FP'])
        _U = float(flask.request.form['_U'])
        data = pandas.DataFrame({'IW':_IW,'IF':_IF,'VW':_VW,'FP':_FP,'U':_U},index=[0])
        data['Q'] = data.IW * data.U
        data = scaler.transform(data.values)
        y_pred = loaded_model.predict(data)

        return render_template('main.html', result = f'Глубина шва: {round(y_pred[0][0], 4)}, Ширина шва: {round(y_pred[0][1], 4)}')

if __name__ == '__main__':
    app.run()